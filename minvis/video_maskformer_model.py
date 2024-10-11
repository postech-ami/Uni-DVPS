# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple
import einops
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher
from mask2former_video.utils.memory import retry_if_cuda_oom

from scipy.optimize import linear_sum_assignment

# from demo_video.predictor import VisualizationDemo
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)



@META_ARCH_REGISTRY.register()
class VideoMaskFormer_frame(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,

        visualization,
        # demo
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference

        self.visualization = visualization
        # self.demo = demo

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        # demo = VisualizationDemo(cfg)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            # wandb visualization
            "visualization": cfg.OUTPUT_DIR,
            # "demo": demo
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # if self.visualization:
            #     raw_shape = [batched_inputs[0]["height"], batched_inputs[0]["width"]]
            #     self.visualize(images, outputs, raw_shape)

            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses, None
        else:
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )

        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})

        return outputs, gt_instances

    def match_from_embds(self, tgt_embds, cur_embds):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing(self, outputs):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        # pred_logits: 1 t q c
        # pred_masks: 1 q t h w
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        # make frame list of each output
        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        # append first frame outputs
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        for i in range(1, len(pred_logits)):
            # query matching
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            # sorting by indices
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        # mean logits for one video
        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                # gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            # invaild mean all of ids in videos are -1 (e.g., [-1, -1])
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def panoptic_inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        scores, labels = F.softmax(pred_cls, dim=-1).max(-1)
        pred_masks = pred_masks.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = pred_masks[keep]
        cur_mask_cls = pred_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        # bilinear interpolation
        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, : img_size[0], : img_size[1]]
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        # cur_masks = cur_masks > 0.
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks

        t, h, w = cur_masks.shape[-3:]
        panoptic_seg = torch.zeros((t, h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_ids
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )


            return panoptic_seg, segments_info


    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    def visualize(self, images, seg_outputs, raw_shape):
        # deepcopy
        vis_seg = dict()
        vis_images = copy.deepcopy(images.tensor.detach().cpu())
        vis_seg["pred_logits"] = copy.deepcopy(seg_outputs["pred_logits"].detach())
        vis_seg["pred_masks"] = copy.deepcopy(seg_outputs["pred_masks"].detach())
        vis_seg["pred_embds"] = copy.deepcopy(seg_outputs["pred_embds"].detach())

        outputs = self.post_processing(vis_seg)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]
        first_resize_size = (vis_images.shape[-2], vis_images.shape[-1])

        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        height, width = raw_shape  # raw image size before data augmentation

        predictions = retry_if_cuda_oom(self.inference_video)(mask_cls_result,
                                                              mask_pred_result,
                                                              image_size, height, width,
                                                              first_resize_size)

        total_vis_output = self.demo.process_predictions(vis_images, predictions)
        total_vis_output = np.asarray([vis_output.get_image() for vis_output in total_vis_output])





