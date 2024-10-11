import logging
from typing import Tuple
import einops
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class UniDVPS(nn.Module):
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
        num_frames,
        window_inference,
        visualization,
        dataset
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.num_frames = num_frames
        self.window_inference = window_inference
        self.visualization = visualization
        self.dataset = dataset

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": None,
            "num_queries": cfg.MODEL.UNIFIED_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.UNIFIED_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.UNIFIED_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.UNIFIED_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.UNIFIED_FORMER.TEST.WINDOW_INFERENCE,
            "visualization": cfg.OUTPUT_DIR,
            "dataset": cfg.DATASETS.TRAIN
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
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

        predictions = {}
        if outputs["seg_preds"] is not None:
            outputs_seg = outputs["seg_preds"]
            outputs_seg = self.post_processing(outputs_seg)

            mask_cls_results = outputs_seg["pred_logits"]
            mask_pred_results = outputs_seg["pred_masks"]
            mask_embds_results = outputs_seg['pred_embds']


            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            outputs_depth = outputs["depth_preds"]["pred_depths"][0]
            global_depth = outputs["depth_preds"]["pred_global_depths"][-1]
            panoptic_seg, segments_info, depth_output, depth_hole = (retry_if_cuda_oom(self.panoptic_depth_inference_video)
                                                                     (outputs_depth, global_depth, mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, mask_embds_results))
            predictions["seg_output"] = (panoptic_seg, segments_info)
            predictions["depth_output"] = depth_output
            return predictions

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
        outputs['pred_embds'] = torch.stack(out_embds, dim=0)

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
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['seg_preds']['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['seg_preds']['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['seg_preds']['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def prepare_targets_seg(self, targets, images):
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

    def prepare_targets_depth(self, targets, images, targets_seg):
        h_pad, w_pad = images.tensor.shape[-2:]

        gt_depths = []
        for v_i, targets_per_video in enumerate(targets):
            depth_shape = [self.num_frames, h_pad, w_pad]
            gt_depth_per_video = torch.zeros(depth_shape, device=self.device)

            for f_i, targets_per_frame in enumerate(targets_per_video["depth"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.shape
                gt_depth_per_video[f_i, :h, :w] = targets_per_frame

            gt_masks_per_video = targets_seg[v_i]['masks']
            num_instance = len(gt_masks_per_video)
            gt_depths.append({"depth": gt_depth_per_video})

        return gt_depths

    def panoptic_depth_inference_video(self, pred_depth, global_depth, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size, pred_embds):
        scores, labels = F.softmax(pred_cls, dim=-1).max(-1)
        # scores, labels = F.softmax(pred_cls[:,:-1], dim=-1).max(-1)
        pred_masks = pred_masks.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = pred_masks[keep]
        cur_depths = pred_depth[keep]

        cur_masks = F.interpolate(cur_masks, size=first_resize_size, mode="bilinear", align_corners=False)
        cur_masks = cur_masks[:, :, : img_size[0], : img_size[1]]
        cur_masks = F.interpolate(cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False)
        cur_depths = F.interpolate(cur_depths, size=first_resize_size, mode="nearest",)
        cur_depths = cur_depths[:, :, : img_size[0], : img_size[1]]
        cur_depths = F.interpolate(cur_depths, size=(output_height, output_width), mode="nearest",)
        global_depth = F.interpolate(global_depth, size=first_resize_size, mode="nearest",)
        global_depth = global_depth[:, :, : img_size[0], : img_size[1]]
        global_depth = F.interpolate(global_depth, size=(output_height, output_width), mode="nearest",).squeeze(1)
        global_depth = torch.clamp(global_depth, 0.001, 80.)

        visualize_global_depth = global_depth.clone()
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks

        t, h, w = cur_masks.shape[-3:]
        panoptic_seg = torch.zeros((t, h, w), dtype=torch.int32, device=cur_masks.device)
        depth_map = torch.zeros((t, h, w), dtype=cur_depths.dtype, device=cur_depths.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
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
                    global_depth[mask] = cur_depths[k][mask]
                    depth_map[mask] = cur_depths[k][mask]

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

        return panoptic_seg, segments_info, global_depth, depth_map