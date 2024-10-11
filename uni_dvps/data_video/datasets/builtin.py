import os
from .cityscapes_dvps import (
    register_cityscapes_dvps,
    _get_cityscapes_dvps_meta
)
from .semkitti_dvps import (
    register_semkitti_dvps,
    _get_semkitti_dvps_meta
)

# ==== Predefined splits for Cityscpaes-DVPS ===========
_PREDEFINED_SPLITS_CITYSCAPES_DVPS = {
    "cityscapes_dvps_val": (
        "cityscapes-dvps/video_sequence/val",
        "cityscapes-dvps/video_sequence/val",
        "cityscapes-dvps/video_sequence/dvps_cityscapes_val.json",
    ),
}

# ==== Predefined splits for SemKITTI-DVPS ===========
_PREDEFINED_SPLITS_SEM_KITTI = {
    "semkitti_dvps_val": (
        "semkitti-dvps/video_sequence/val",
        "semkitti-dvps/video_sequence/val",
        "semkitti-dvps/video_sequence/dvps_semkitti_val.json"
    ),
}

def register_all_cityscapes_dvps(root):
    for key, (image_dir, gt_dir, gt_json) in _PREDEFINED_SPLITS_CITYSCAPES_DVPS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        register_cityscapes_dvps(
            key,
            _get_cityscapes_dvps_meta(),
            os.path.join(root, gt_json) if "://" not in gt_json else gt_json,
            os.path.join(root, image_dir),
            os.path.join(root, gt_dir),
        )

def register_all_sem_kitti(root):
    for key, (image_dir, gt_dir, gt_json) in _PREDEFINED_SPLITS_SEM_KITTI.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        if "val" in key:
            for eval_frames in [1, 5, 10, 20]:
                new_key = key+str(eval_frames)
                register_semkitti_dvps(
                    new_key,
                    _get_semkitti_dvps_meta(),
                    os.path.join(root, gt_json) if "://" not in gt_json else gt_json,
                    os.path.join(root, image_dir),
                    os.path.join(root, gt_dir),
                )
        else:
            register_semkitti_dvps(
                key,
                _get_semkitti_dvps_meta(),
                os.path.join(root, gt_json) if "://" not in gt_json else gt_json,
                os.path.join(root, image_dir),
                os.path.join(root, gt_dir),
            )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_dvps(_root)
    register_all_sem_kitti(_root)

