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

"""
This file contains functions to parse SemKITTI-DVPS dataset into dicts in "Detectron2 format".
"""
logger = logging.getLogger(__name__)
__all__ = ["load_semkitti_dvps_json", "register_semkitti_dvps"]

SEMKITTI_CATEGORIES = [
    {"color": (245, 150, 100), "isthing": 1, "id": 10, "trainId": 0, "name": "car"},
    {"color": (245, 230, 100), "isthing": 1, "id": 11, "trainId": 1, "name": "bicycle"},
    {"color": (150,  60,  30), "isthing": 1, "id": 15, "trainId": 2, "name": "motorcycle"},
    {"color": (180,  30,  80), "isthing": 1, "id": 18, "trainId": 3, "name": "truck"},
    {"color": (255,   0,   0), "isthing": 1, "id": 20, "trainId": 4, "name": "other-vehicle"},
    {"color": ( 30,  30, 255), "isthing": 1, "id": 30, "trainId": 5, "name": "person"},
    {"color": (200,  40, 255), "isthing": 1, "id": 31, "trainId": 6, "name": "bicyclist"},
    {"color": ( 90,  30, 150), "isthing": 1, "id": 32, "trainId": 7, "name": "motorcyclist"},

    {"color": (255,   0, 255), "isthing": 0, "id": 40, "trainId": 8, "name": "road"},
    {"color": (255, 150, 255), "isthing": 0, "id": 44, "trainId": 9, "name": "parking"},
    {"color": ( 75,   0,  75), "isthing": 0, "id": 48, "trainId": 10, "name": "sidewalk"},
    {"color": ( 75,   0, 175), "isthing": 0, "id": 49, "trainId": 11, "name": "other-ground"},
    {"color": (  0, 200, 255), "isthing": 0, "id": 50, "trainId": 12, "name": "building"},
    {"color": ( 50, 120, 255), "isthing": 0, "id": 51, "trainId": 13, "name": "fence"},
    {"color": (  0, 175,   0), "isthing": 0, "id": 70, "trainId": 14, "name": "vegetation"},
    {"color": (  0,  60, 135), "isthing": 0, "id": 71, "trainId": 15, "name": "trunk"},
    {"color": ( 80, 240, 150), "isthing": 0, "id": 72, "trainId": 16, "name": "terrain"},
    {"color": (150, 240, 255), "isthing": 0, "id": 80, "trainId": 17, "name": "pole"},
    {"color": (  0,   0, 255), "isthing": 0, "id": 81, "trainId": 18, "name": "traffic-sign"},
]

def _get_semkitti_dvps_meta():
    thing_ids = [k["trainId"] for k in SEMKITTI_CATEGORIES if k["isthing"] == 1]
    thing_classes = [k["name"] for k in SEMKITTI_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SEMKITTI_CATEGORIES if k["isthing"] == 1]
    stuff_ids = [k["trainId"] for k in SEMKITTI_CATEGORIES if k["isthing"] == 0]
    stuff_classes = [k["name"] for k in SEMKITTI_CATEGORIES if k["isthing"] == 0]
    stuff_colors = [k["color"] for k in SEMKITTI_CATEGORIES if k["isthing"] == 0]

    assert len(thing_ids) == 8, len(thing_ids)
    assert len(stuff_ids) == 11, len(stuff_ids)

    # Mapping from the incontiguous SEMKITTI_DVPS category id to an id in [0, 10]
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

def load_semkitti_dvps_json(gt_json, image_dir, gt_dir, meta, name):
    assert os.path.exists(gt_json), gt_json+" not exists"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    dataset_dicts = []
    if 'train' in name:
        for vid in file_dicts.keys():
            for fid in file_dicts[vid].keys():
                if fid == "000000":
                    record = {}
                    record["video_id"] = vid
                    record["height"] = file_dicts[vid][fid]["height"]
                    record["width"] = file_dicts[vid][fid]["width"]
                    record["file_names"] = [os.path.join(image_dir, file_dicts[vid][fid]["image"])]
                    record["class_file_names"] = [os.path.join(image_dir, file_dicts[vid][fid]["class"])]
                    record["instance_file_names"] = [os.path.join(image_dir, file_dicts[vid][fid]["instance"])]
                    record["depth_file_names"] = [os.path.join(image_dir, file_dicts[vid][fid]["depth"])]
                    dataset_dicts.append(record)
                else:
                    dataset_dicts[-1]["file_names"].append(os.path.join(image_dir, file_dicts[vid][fid]["image"]))
                    dataset_dicts[-1]["class_file_names"].append(os.path.join(image_dir, file_dicts[vid][fid]["class"]))
                    dataset_dicts[-1]["instance_file_names"].append(os.path.join(image_dir, file_dicts[vid][fid]["instance"]))
                    dataset_dicts[-1]["depth_file_names"].append(os.path.join(image_dir, file_dicts[vid][fid]["depth"]))

    elif 'val' in name:
        len_vid = int(name.split('val')[1])
        for vid in file_dicts.keys():
            for fid in file_dicts[vid].keys():
                if int(fid)+len_vid > len(file_dicts[vid]):
                    continue
                for i in range(len_vid):
                    if i == 0:
                        record = {}
                        record["video_id"] = vid
                        record["height"] = file_dicts[vid][fid]["height"]
                        record["width"] = file_dicts[vid][fid]["width"]
                        record["file_names"] = [os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][fid]["image"])]
                        record["class_file_names"] = [os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][fid]["class"])]
                        record["instance_file_names"] = [os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][fid]["instance"])]
                        record["depth_file_names"] = [os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][fid]["depth"])]
                        dataset_dicts.append(record)
                        i += 1
                    else:
                        next_fid = '{0:06d}'.format(int(fid)+i)
                        dataset_dicts[-1]["file_names"].append(os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][next_fid]["image"]))
                        dataset_dicts[-1]["class_file_names"].append(os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][next_fid]["class"]))
                        dataset_dicts[-1]["instance_file_names"].append(os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][next_fid]["instance"]))
                        dataset_dicts[-1]["depth_file_names"].append(os.path.join(image_dir, str(fid)+'_'+file_dicts[vid][next_fid]["depth"]))
                        i += 1


    # logger.info("Loaded {} images from {}".format(len(file_dicts), image_dir))
    return dataset_dicts


def register_semkitti_dvps(name, meta, gt_json, image_dir, gt_dir):
    """
    Register a dataset in Cityscapes_DVPS's json annotation format for DVPS.
    """
    assert isinstance(name, str), name
    assert isinstance(gt_json, (str, os.PathLike)), gt_json
    assert isinstance(image_dir, (str, os.PathLike)), image_dir
    assert isinstance(gt_dir, (str, os.PathLike)), gt_dir

    DatasetCatalog.register(name, lambda: load_semkitti_dvps_json(gt_json, image_dir, gt_dir, meta, name))
    MetadataCatalog.get(name).set(
        panoptic_root=gt_dir,
        image_root=image_dir,
        gt_dir=gt_dir,
        evaluator_type="semkitti_dvps",
        ignore_label=255,
        **meta,
    )
