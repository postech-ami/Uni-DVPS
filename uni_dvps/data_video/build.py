# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from typing import Optional
import os
import torch
import time

from detectron2.engine import hooks
from detectron2.utils import comm
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.utils.events import (
    EventWriter,
    get_event_storage,
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter
)


def build_hooks(self):
    cfg = self.cfg.clone()
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

    ret = [
        hooks.IterationTimer(),
        hooks.LRScheduler(),
        hooks.PreciseBN(
            # Run at the same freq as (but before) evaluation.
            cfg.TEST.EVAL_PERIOD,
            self.model,
            # Build a new data loader to not affect training
            self.build_train_loader(cfg),
            cfg.TEST.PRECISE_BN.NUM_ITER,
        )
        if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
        else None,
    ]

    # Do PreciseBN before checkpointer, because it updates the model and need to
    # be saved by checkpointer.
    # This is not always the best: if checkpointing has a different frequency,
    # some checkpoints may have more precise statistics than others.
    if comm.is_main_process():
        ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

    def test_and_save_results():
        self._last_eval_results = self.test(self.cfg, self.model)
        return self._last_eval_results

    # Do evaluation after checkpointer, because then if it fails,
    # we can use the saved checkpoint to debug.
    ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

    if comm.is_main_process():
        # Here the default print/log frequency of each writer is used.
        # run writers in the end, so that evaluation metrics are written
        if cfg.OUTPUT_DIR:
            ret.append(hooks.PeriodicWriter(build_writers(cfg.OUTPUT_DIR, self.max_iter), period=cfg.TEST.LOG_PERIOD))
    return ret

def build_writers(output_dir: str, max_iter: Optional[int] = None):
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        # TensorboardXWriter(output_dir),
        # WAndBWriter()
    ]

def run_step(self):
    """
    Implement the AMP training logic.
    """
    assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
    assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
    from torch.cuda.amp import autocast

    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start

    with autocast():
        loss_dict, image = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

    self.optimizer.zero_grad()
    # depth.retain_grad()
    self.grad_scaler.scale(losses).backward()

    self._write_metrics(loss_dict, data_time)

    if isinstance(image, torch.Tensor):
        _log_images(image)

    self.grad_scaler.step(self.optimizer)
    self.grad_scaler.update()

def _log_images(image):
    image_name = "depth"
    if comm.is_main_process():
        storage = get_event_storage()
        storage.put_image(image_name, image)