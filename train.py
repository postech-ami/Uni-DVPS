"""
Uni-DVPS Training Script.
This script is based on Mask2Former and MinVIS.
"""
import os
import copy
import itertools
import logging
import torch
from collections import OrderedDict
from typing import Any, Dict, List, Set

# detectron2
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# models
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import (
    add_minvis_config,
    build_detection_test_loader,
)
from uni_dvps import (
    add_uni_dvps_config,
    CityscapesDVPSDatasetMapper,
    CityscapesDVPSEvaluator,
    SemkittiDVPSDatasetMapper,
    SemkittiDVPSEvaluator,
)

import warnings
warnings.filterwarnings(action='ignore')


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        if "cityscapes" in dataset_name:
            evaluator = CityscapesDVPSEvaluator(dataset_name, output_folder)
        if "kitti" in dataset_name:
            evaluator = SemkittiDVPSEvaluator(dataset_name, output_folder, eval_frame=int(dataset_name.split('val')[1]))

        return evaluator

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        if "cityscapes" in dataset_name:
            mapper = CityscapesDVPSDatasetMapper(cfg, is_train=False)
        if "kitti" in dataset_name:
            mapper = SemkittiDVPSDatasetMapper(cfg, is_train= False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def test(cls, cfg, model, evaluators=None, eval_frames=None):
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_uni_dvps_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.EVAL_FRAMES:
        cfg.DATASETS.TEST = (cfg.DATASETS.TEST[0]+str(cfg.EVAL_FRAMES),)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="uni_dvps")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
