import os
import random
import copy
import torch
import numpy as np
from typing import List, Union
from PIL import Image
import logging

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import pycocotools

from .augmentation import build_augmentation, build_semkitti_augmentation

def _get_dummy_info(ignore_label):
    return {
        "id": -1,
        "category_id": ignore_label,
        "area": 0,
        "bbox": np.array([0, 0, 0, 0]),
        "iscrowd": 0
    }


class CityscapesDVPSDatasetMapper:
    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            sampling_frame_num: int = 2,
            sampling_frame_range: int = 6,
            sampling_frame_shuffle: bool = False,
            sampling_frame_ratio: float = 1.0,
            size_divisibility: int = 32,
            num_classes: int = 19,
            depth_bound: bool = True,
    ):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.sampling_frame_ratio = sampling_frame_ratio
        self.size_divisibility = size_divisibility
        self.num_classes = num_classes
        self.depth_bound = depth_bound

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": cfg.INPUT.SAMPLING_FRAME_NUM,
            "sampling_frame_range": cfg.INPUT.SAMPLING_FRAME_RANGE,
            "sampling_frame_shuffle": cfg.INPUT.SAMPLING_FRAME_RANGE,
            "sampling_frame_ratio": cfg.INPUT.SAMPLING_FRAME_RATIO,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "depth_bound": cfg.INPUT.DEPTH_BOUND
        }
        return ret

    def select_frames(self, video_length):
        if self.sampling_frame_ratio < 1.0:
            assert self.sampling_frame_num == 1, "only support subsampling for a single frame"
            subsampled_frames = max(int(np.round(video_length * self.sampling_frame_ratio)), 1)
            if subsampled_frames > 1:
                # deterministic uniform subsampling given video length
                subsampled_idx = np.linspace(0, video_length, num=subsampled_frames, endpoint=False, dtype=int)
                ref_idx = random.randrange(subsampled_frames)
                ref_frame = subsampled_idx[ref_idx]
            else:
                ref_frame = video_length // 2  # middle frame

            selected_idx = [ref_frame]
        else:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)

        return selected_idx

    def get_segments_info(self, original_format):
        # visualization
        pan_color_format = np.zeros((original_format.shape[0], original_format.shape[1],3), dtype=np.uint8)

        segmentIds = np.unique(original_format)
        segmInfo = []
        for segmentId in segmentIds:
            if segmentId < 1000:
                categoryId = segmentId
                iscrowd = 1
            else:
                categoryId = segmentId // 1000
                iscrowd = 0
            if categoryId == 32: # ignore label
                continue
            if categoryId < 11:
                iscrowd = 0

            mask = original_format == segmentId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            pan_color_format[mask] = color

            # area computation
            area = np.sum(mask)
            #bbox computation
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]

            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": iscrowd })

        return segmInfo

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        video_length = len(dataset_dict["file_names"])
        if self.is_train:
            selected_idx = self.select_frames(video_length)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        file_names = dataset_dict.pop("file_names", None)
        seg_file_names = dataset_dict.pop("seg_file_names", None)
        depth_file_names = dataset_dict.pop("depth_file_names", None)

        if self.is_train:
            _ids, ids = set(), dict()
            for frame_idx in selected_idx:
                _pan_seg_gt = np.array(Image.open(seg_file_names[frame_idx]))
                _segments_info = self.get_segments_info(_pan_seg_gt)
                _ids.update([segment["id"] for segment in _segments_info])
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        dataset_dict["depth"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            if seg_file_names:
                pan_seg_gt = np.array(Image.open(seg_file_names[frame_idx]))
                segments_info = self.get_segments_info(pan_seg_gt)
            else:
                pan_seg_gt, segments_info = None, None
            if depth_file_names:
                depth_gt = np.array(Image.open(depth_file_names[frame_idx]))
                depth_gt_1 = (depth_gt // 256).astype(np.uint8)
                depth_gt_2 = (depth_gt % 256).astype(np.uint8)
            else:
                depth_gt = None

            if pan_seg_gt is None:
                raise ValueError(
                    "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                        seg_file_names
                    )
                )
            if depth_gt is None:
                raise ValueError(
                    "Cannot find 'depth_file_name' for panoptic segmentation dataset {}.".format(
                        depth_file_names
                    )
                )

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image.copy()

            # apply same transformation to panoptic segmentation
            from panopticapi.utils import rgb2id, id2rgb

            pan_seg_gt = id2rgb(pan_seg_gt)
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            pan_seg_gt = rgb2id(pan_seg_gt)


            if image.shape[:-1] != (1024, 2048):
                print(image.shape)
            template = np.ones((1024, 2048))
            template = transforms.apply_segmentation(template)
            image[template == 0] = 128
            pan_seg_gt[template == 0] = 32000

            # apply same transformation to depth
            depth_gt_1 = transforms.apply_segmentation(depth_gt_1)
            depth_gt_2 = transforms.apply_segmentation(depth_gt_2)
            depth_gt = depth_gt_1.astype(np.float64) * 256 + depth_gt_2.astype(np.float64)
            depth_gt = depth_gt / 256.
            del depth_gt_1, depth_gt_2

            depth_gt[template == 0] = 0

            # augmentation for depth bound
            for transform in transforms:
                if isinstance(transform, T.ResizeTransform):
                    aug_scale = (transform.w / transform.new_w + transform.h / transform.new_h) / 2
                    if self.depth_bound:
                        depth_gt = np.clip(depth_gt * aug_scale, depth_gt.min(), depth_gt.max())
                    else:
                        depth_gt = depth_gt * aug_scale

            # Pad image and segmentation label
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
            depth_gt = torch.as_tensor(np.ascontiguousarray(depth_gt).astype("float32"))
            dataset_dict["depth"].append(depth_gt)

            image_shape = (image.shape[-2], image.shape[-1])  # h, w
            dataset_dict["image"].append(image)

            if not self.is_train:
                continue

            # make video segments_info
            video_segments_info = [_get_dummy_info(32) for _ in range(len(ids))]
            for segment_info in segments_info:
                video_segments_info[ids[segment_info["id"]]] = segment_info

            # create Instance object from video segments_info
            instances = Instances(image_shape)
            _gt_classes, _gt_ids, _gt_masks = [], [], []
            pan_seg_gt = pan_seg_gt.numpy()
            for segment_info in video_segments_info:
                if not segment_info["iscrowd"]:
                    _gt_classes.append(segment_info["category_id"])
                    _gt_ids.append(segment_info["id"])
                    _gt_masks.append(pan_seg_gt == segment_info["id"])

            instances.gt_classes = torch.tensor(_gt_classes, dtype=torch.int64)
            instances.gt_ids = torch.as_tensor(_gt_ids)

            if len(_gt_masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _gt_masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"].append(instances)

        return dataset_dict


class SemkittiDVPSDatasetMapper:
    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            sampling_frame_num: int = 2,
            sampling_frame_range: int = 6,
            sampling_frame_shuffle: bool = False,
            sampling_frame_ratio: float = 1.0,
            size_divisibility: int = 32,
            num_classes: int = 19,
            depth_bound: bool = True,
    ):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.sampling_frame_ratio = sampling_frame_ratio
        self.size_divisibility = size_divisibility
        self.num_classes = num_classes
        self.depth_bound = depth_bound

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_semkitti_augmentation(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": cfg.INPUT.SAMPLING_FRAME_NUM,
            "sampling_frame_range": cfg.INPUT.SAMPLING_FRAME_RANGE,
            "sampling_frame_shuffle": cfg.INPUT.SAMPLING_FRAME_RANGE,
            "sampling_frame_ratio": cfg.INPUT.SAMPLING_FRAME_RATIO,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "depth_bound": cfg.INPUT.DEPTH_BOUND
        }
        return ret

    def select_frames(self, video_length):
        if self.sampling_frame_ratio < 1.0:
            assert self.sampling_frame_num == 1, "only support subsampling for a single frame"
            subsampled_frames = max(int(np.round(video_length * self.sampling_frame_ratio)), 1)
            if subsampled_frames > 1:
                # deterministic uniform subsampling given video length
                subsampled_idx = np.linspace(0, video_length, num=subsampled_frames, endpoint=False, dtype=int)
                ref_idx = random.randrange(subsampled_frames)
                ref_frame = subsampled_idx[ref_idx]
            else:
                ref_frame = video_length // 2  # middle frame

            selected_idx = [ref_frame]
        else:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)

        return selected_idx


    def get_segments_info(self, class_gt, instance_gt):
        assert instance_gt.max() < 1000, "instance_gt is over 1000"

        pan_seg_gt = class_gt * 1000 + instance_gt
        segmentIds = np.unique(pan_seg_gt)
        segmInfo = []
        for segmentId in segmentIds:
            categoryId = segmentId // 1000
            iscrowd = 0
            if categoryId == 255:
                continue

            mask = pan_seg_gt == segmentId
            area = np.sum(mask)

            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area)})
        return segmInfo, pan_seg_gt


    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        video_length = len(dataset_dict["file_names"])
        if self.is_train:
            selected_idx = self.select_frames(video_length)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        file_names = dataset_dict.pop("file_names", None)
        class_file_names = dataset_dict.pop("class_file_names", None)
        instance_file_names = dataset_dict.pop("instance_file_names", None)
        depth_file_names = dataset_dict.pop("depth_file_names", None)

        if not self.is_train:
            # remove first frame index of batch in file names
            batch_idx = os.path.basename(file_names[0]).split('_')[0] + "_"
            file_names = [name.replace(batch_idx, "", 1) for name in file_names]
            class_file_names = [name.replace(batch_idx, "", 1) for name in class_file_names]
            instance_file_names = [name.replace(batch_idx, "", 1) for name in instance_file_names]
            depth_file_names = [name.replace(batch_idx, "", 1) for name in depth_file_names]

        if self.is_train:
            _ids, ids = set(), dict()
            for frame_idx in selected_idx:
                _class_gt = np.array(Image.open(class_file_names[frame_idx]))
                _instance_gt = np.array(Image.open(instance_file_names[frame_idx]))
                _segments_info, _ = self.get_segments_info(_class_gt, _instance_gt)
                _ids.update([segment["id"] for segment in _segments_info])
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        dataset_dict["depth"] = []
        if not self.is_train:
            dataset_dict["batch_idx"] = batch_idx

        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            ori_image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, ori_image)

            if class_file_names and instance_file_names:
                class_gt = np.array(Image.open(class_file_names[frame_idx]))
                instance_gt = np.array(Image.open(instance_file_names[frame_idx]))
                segments_info, pan_seg_gt = self.get_segments_info(class_gt, instance_gt)
            else:
                segments_info= None
            if depth_file_names:
                depth_gt = np.array(Image.open(depth_file_names[frame_idx]))
                depth_gt_1 = (depth_gt // 256).astype(np.uint8)
                depth_gt_2 = (depth_gt % 256).astype(np.uint8)
            else:
                depth_gt = None

            if segments_info is None:
                raise ValueError(
                    "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(class_file_names)
                )
            if depth_gt is None:
                raise ValueError(
                    "Cannot find 'depth_file_name' for panoptic segmentation dataset {}.".format(depth_file_names)
                )

            aug_input = T.AugInput(ori_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image.copy()

            from panopticapi.utils import rgb2id, id2rgb
            pan_seg_gt = id2rgb(pan_seg_gt)
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            pan_seg_gt = rgb2id(pan_seg_gt)
            pan_seg_gt[pan_seg_gt==0] = 255000

            template = np.ones(instance_gt.shape)
            template = transforms.apply_segmentation(template)
            image[template == 0] = 128
            pan_seg_gt[template == 0] = 255000

            # apply same transformation to depth
            depth_gt_1 = transforms.apply_segmentation(depth_gt_1)
            depth_gt_2 = transforms.apply_segmentation(depth_gt_2)
            depth_gt = depth_gt_1.astype(np.float64) * 256 + depth_gt_2.astype(np.float64)
            depth_gt = depth_gt / 256.
            del depth_gt_1, depth_gt_2
            depth_gt[template == 0] = 0

            # augmentation for depth bound
            for transform in transforms:
                if isinstance(transform, T.ResizeTransform):
                    aug_scale = (transform.w / transform.new_w + transform.h / transform.new_h) / 2
                    if self.depth_bound:
                        depth_gt = np.clip(depth_gt * aug_scale, depth_gt.min(), depth_gt.max())
                    else:
                        depth_gt = depth_gt * aug_scale

            # Pad image and segmentation label
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            image_shape = (image.shape[-2], image.shape[-1])  # h, w
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
            depth_gt = torch.as_tensor(np.ascontiguousarray(depth_gt).astype("float32"))
            dataset_dict["depth"].append(depth_gt)
            dataset_dict["image"].append(image)

            if not self.is_train:
                continue

            # make video segments_info
            video_segments_info = [_get_dummy_info(255) for _ in range(len(ids))]
            for segment_info in segments_info:
                # class_id = segment_info["category_id"]
                video_segments_info[ids[segment_info["id"]]] = segment_info

            # create Instance object from video segments_info
            instances = Instances(image_shape)
            _gt_classes, _gt_ids, _gt_masks = [], [], []
            pan_seg_gt = pan_seg_gt.numpy()
            for segment_info in video_segments_info:
                _gt_classes.append(segment_info["category_id"])
                _gt_ids.append(segment_info["id"])
                _gt_masks.append(pan_seg_gt == segment_info["id"])

            instances.gt_classes = torch.tensor(_gt_classes, dtype=torch.int64)
            instances.gt_ids = torch.as_tensor(_gt_ids)

            if len(_gt_masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _gt_masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"].append(instances)

        return dataset_dict