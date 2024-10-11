import os
import numpy as np
import logging
import torch
from collections import OrderedDict
from PIL import Image
from tabulate import tabulate
import tempfile

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.utils import comm
from . import eval_dvpq_cityscapes, eval_dvpq_semkitti

logger = logging.getLogger(__name__)


class CityscapesDVPSEvaluator(CityscapesEvaluator):
    def __init__(self, dataset_name, output_folder):
        super().__init__(dataset_name)
        self.output_folder = output_folder
        self.eval_frames = [1, 2, 3, 4]
        self.depth_thres = [0.5, 0.25, 0.1]
        self.ign_id = 19

    def process(self, inputs, outputs):
        save_dir = self._temp_dir
        panoptic_seg, segments_info = outputs['seg_output']

        for f_i, pred_frame in enumerate(panoptic_seg):
            instance_seg = pred_frame.detach().cpu().numpy()
            semantic_seg = np.zeros_like(instance_seg) + self.ign_id
            for s_info in segments_info:
                semantic_seg[instance_seg == s_info['id']] = s_info['category_id']
                if not s_info['isthing']:
                    stuff_segm = instance_seg == s_info['id']
                    instance_seg[stuff_segm] = 0
            aggregate_result = np.stack(
                [semantic_seg, instance_seg, np.zeros_like(instance_seg)], axis=2
            ).astype(np.uint8)

            basename = os.path.basename(inputs[0]['file_names'][f_i])
            filename = os.path.join(save_dir, basename)
            Image.fromarray(aggregate_result).save(filename.replace("_leftImg8bit.png", "_panoptic.png"))

        if outputs['depth_output'] is not None:
            depth = outputs['depth_output']
            for f_i, pred_depth_frame in enumerate(depth):
                pred_depth = pred_depth_frame.detach().cpu().numpy()
                pred_depth = (pred_depth*256).astype(np.int32)
                basename = os.path.basename(inputs[0]['file_names'][f_i])
                filename = os.path.join(save_dir, basename)
                Image.fromarray(pred_depth).save(filename.replace("_leftImg8bit.png", "_depth.png"))


    def evaluate(self):
        comm.synchronize()
        if not comm.is_main_process():
            return

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        dvpq = dict()
        pred_dir = self._temp_dir
        gt_dir = os.path.join(os.environ['DETECTRON2_DATASETS'], self._metadata.gt_dir)
        for depth_thres in self.depth_thres:
            dvpq[depth_thres] = dict()
            for eval_frame in self.eval_frames:
                results = eval_dvpq_cityscapes.main(eval_frame, pred_dir, gt_dir, depth_thres)
                dvpq[depth_thres][eval_frame] = {'dvpq': results['averages'][0],
                                                 'dvpq_th': results['averages'][1],
                                                 'dvpq_st': results['averages'][2]}

        ret = OrderedDict()
        for depth_thres in self.depth_thres:
            ret[f"lambda={depth_thres}"] = {
                'dvpq': np.array([dvpq[depth_thres][eval_frame]['dvpq'] for eval_frame in self.eval_frames]).mean(),
                'dvpq_th': np.array([dvpq[depth_thres][eval_frame]['dvpq_th'] for eval_frame in self.eval_frames]).mean(),
                'dvpq_st': np.array([dvpq[depth_thres][eval_frame]['dvpq_st'] for eval_frame in self.eval_frames]).mean()
            }

        for eval_frame in self.eval_frames:
            ret[f"k={eval_frame}"] ={
                'dvpq': np.array([dvpq[depth_thres][eval_frame]['dvpq'] for depth_thres in self.depth_thres]).mean(),
                'dvpq_th': np.array([dvpq[depth_thres][eval_frame]['dvpq_th'] for depth_thres in self.depth_thres]).mean(),
                'dvpq_st': np.array([dvpq[depth_thres][eval_frame]['dvpq_st'] for depth_thres in self.depth_thres]).mean(),
            }

        ret['average'] = {
            'dvpq': np.array([ret[f"lambda={depth_thres}"]['dvpq'] for depth_thres in self.depth_thres]).mean(),
            'dvpq_th': np.array([ret[f"lambda={depth_thres}"]['dvpq_th'] for depth_thres in self.depth_thres]).mean(),
            'dvpq_st': np.array([ret[f"lambda={depth_thres}"]['dvpq_st'] for depth_thres in self.depth_thres]).mean(),
        }

        self._working_dir.cleanup()
        return ret


class SemkittiDVPSEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_folder, eval_frame):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.output_folder = output_folder
        # self.eval_frames = [1, 5, 10, 20]
        self.eval_frames = [eval_frame]
        self.depth_thres = [0.5, 0.25, 0.1]
        self.ign_id = 19

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="semkitti_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing semkitti results to temporary directory {} ...".format(self._temp_dir)
        )

    def process(self, inputs, outputs):
        save_dir = self._temp_dir
        # save_dir = '/local_data2/ryeon/minvis_depth/semkitti_val_result'
        panoptic_seg, segments_info = outputs['seg_output']

        for f_i, pred_frame in enumerate(panoptic_seg):
            instance_seg = pred_frame.detach().cpu().numpy()
            semantic_seg = np.zeros_like(instance_seg) + self.ign_id
            for s_info in segments_info:
                semantic_seg[instance_seg == s_info['id']] = s_info['category_id']
                if not s_info['isthing']:
                    stuff_segm = instance_seg == s_info['id']
                    instance_seg[stuff_segm] = 0
            # aggregate_result = np.stack(
            #     [semantic_seg, instance_seg, np.zeros_like(instance_seg)], axis=2
            # ).astype(np.uint8)

            basename = os.path.basename(inputs[0]['file_names'][f_i])
            batch_idx = inputs[0]['batch_idx']
            filename = os.path.join(save_dir, batch_idx+basename)

            Image.fromarray(semantic_seg.astype(np.uint8)).save(filename.replace("_leftImg8bit.png", "_cat.png"))
            Image.fromarray(instance_seg.astype(np.uint8)).save(filename.replace("_leftImg8bit.png", "_ins.png"))
            # Image.fromarray(aggregate_result).save(filename.replace("_leftImg8bit.png", "_panoptic.png"))

        if outputs['depth_output'] is not None:
            depth = outputs['depth_output']
            for f_i, pred_depth_frame in enumerate(depth):
                pred_depth = pred_depth_frame.detach().cpu().numpy()
                pred_depth = (pred_depth * 256).astype(np.int32)
                basename = os.path.basename(inputs[0]['file_names'][f_i])
                batch_idx = inputs[0]['batch_idx']
                filename = os.path.join(save_dir, batch_idx+basename)
                Image.fromarray(pred_depth).save(filename.replace("_leftImg8bit.png", "_depth.png"))

    def evaluate(self):
        comm.synchronize()
        if not comm.is_main_process():
            return

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        dvpq = dict()
        pred_dir = self._temp_dir
        gt_dir = os.path.join(os.environ['DETECTRON2_DATASETS'], self._metadata.gt_dir)
        for depth_thres in self.depth_thres:
            dvpq[depth_thres] = dict()
            for eval_frame in self.eval_frames:
                results = eval_dvpq_semkitti.main(eval_frame, pred_dir, gt_dir, depth_thres)
                dvpq[depth_thres][eval_frame] = {'dvpq': results['averages'][0],
                                                 'dvpq_th': results['averages'][1],
                                                 'dvpq_st': results['averages'][2]}
        ret = OrderedDict()
        for depth_thres in self.depth_thres:
            ret[f"lambda={depth_thres}"] = {
                'dvpq': np.array([dvpq[depth_thres][eval_frame]['dvpq'] for eval_frame in self.eval_frames]).mean(),
                'dvpq_th': np.array([dvpq[depth_thres][eval_frame]['dvpq_th'] for eval_frame in self.eval_frames]).mean(),
                'dvpq_st': np.array([dvpq[depth_thres][eval_frame]['dvpq_st'] for eval_frame in self.eval_frames]).mean()
            }

        for eval_frame in self.eval_frames:
            ret[f"k={eval_frame}"] ={
                'dvpq': np.array([dvpq[depth_thres][eval_frame]['dvpq'] for depth_thres in self.depth_thres]).mean(),
                'dvpq_th': np.array([dvpq[depth_thres][eval_frame]['dvpq_th'] for depth_thres in self.depth_thres]).mean(),
                'dvpq_st': np.array([dvpq[depth_thres][eval_frame]['dvpq_st'] for depth_thres in self.depth_thres]).mean(),
            }

        ret['average'] = {
            'dvpq': np.array([ret[f"lambda={depth_thres}"]['dvpq'] for depth_thres in self.depth_thres]).mean(),
            'dvpq_th': np.array([ret[f"lambda={depth_thres}"]['dvpq_th'] for depth_thres in self.depth_thres]).mean(),
            'dvpq_st': np.array([ret[f"lambda={depth_thres}"]['dvpq_st'] for depth_thres in self.depth_thres]).mean(),
        }
        # _print_depth_results(ret['abs_rel'], ret['rmse'])

        self._working_dir.cleanup()
        return ret


    def _print_depth_results(self, abs_rel, rmse):
        headers = ["ABS_REL", "RMSE"]
        data = []
        row = [abs_rel, rmse]
        data.append(row)

        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".4f", stralign="center", numalign="center"
        )
        logger.info("Depth Evaluation Results:\n" + table)