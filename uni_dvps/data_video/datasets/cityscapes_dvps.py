import contextlib
import io
import json
import logging
import numpy as np
import os
import tqdm
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES

"""
This file contains functions to parse Cityscapes_DVPS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_cityscapes_dvps_json", "register_cityscapes_dvps"]

def _get_cityscapes_dvps_meta():
    thing_ids = [k["trainId"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    stuff_ids = [k["trainId"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 0]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 0]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 0]
    assert len(thing_ids) == 8, len(thing_ids)
    assert len(stuff_ids) == 11, len(stuff_ids)
    # Mapping from the incontiguous Cityscapes_DVPS category id to an id in [0, 10]
    thing_train_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    stuff_train_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}

    ret = {
        "thing_ids": thing_ids,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "thing_train_id_to_contiguous_id": thing_train_id_to_contiguous_id,
        "stuff_ids": stuff_ids,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "stuff_train_id_to_contiguous_id": stuff_train_id_to_contiguous_id
    }
    return ret

def load_cityscapes_dvps_json(gt_json, image_dir, gt_dir, meta, name):
    assert os.path.exists(gt_json), gt_json+" not exists"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    dataset_dicts = []
    for file_dict in file_dicts:
        if file_dict["image"].split("_")[1] == "000000":
            record = {}
            record["height"] = file_dict["height"]
            record["width"] = file_dict["width"]
            # record["length"] = 6
            record["video_id"] = file_dict["image"].split("_")[0]
            record["file_names"] = [os.path.join(image_dir, file_dict["image"])]
            record["seg_file_names"] = [os.path.join(gt_dir, file_dict["seg"])]
            record["depth_file_names"] = [os.path.join(gt_dir, file_dict["depth"])]

            dataset_dicts.append(record)
        else:
            video_id = file_dict["image"].split("_")[0]
            image_name = os.path.join(image_dir, file_dict["image"])
            seg_gt_name = os.path.join(gt_dir, file_dict["seg"])
            depth_gt_name = os.path.join(gt_dir, file_dict["depth"])
            # video_idx = [i for i, dict in enumerate(dataset_dicts) if dict["video_id"] == video_id][0]

            video_idx = int(video_id)
            dataset_dicts[video_idx]["file_names"].append(image_name)
            dataset_dicts[video_idx]["seg_file_names"].append(seg_gt_name)
            dataset_dicts[video_idx]["depth_file_names"].append(depth_gt_name)

    logger.info("Loaded {} images from {}".format(len(file_dicts), image_dir))
    return dataset_dicts


def register_cityscapes_dvps(name, meta, gt_json, image_dir, gt_dir):
    """
    Register a dataset in Cityscapes_DVPS's json annotation format for DVPS.
    """
    assert isinstance(name, str), name
    assert isinstance(gt_json, (str, os.PathLike)), gt_json
    assert isinstance(image_dir, (str, os.PathLike)), image_dir
    assert isinstance(gt_dir, (str, os.PathLike)), gt_dir

    DatasetCatalog.register(name, lambda: load_cityscapes_dvps_json(gt_json, image_dir, gt_dir, meta, name))
    MetadataCatalog.get(name).set(
        panoptic_root=gt_dir,
        image_root=image_dir,
        gt_dir=gt_dir,
        evaluator_type="cityscapes_dvps",
        # ignore_label=255,
        ignore_label=32,
        label_divisor=1000,
        **meta,
    )
