import os
import datetime
from detectron2.config import CfgNode as CN


def add_uni_dvps_config(cfg):
    cfg.INPUT.ROTATE_AUG = 0.
    cfg.INPUT.DEPTH_BOUND = True

    cfg.OUTPUT_DIR = './output'
    cfg.TEST.LOG_PERIOD = 1
    cfg.SEED = 42
    cfg.EVAL_FRAMES = 0

    #PanopticDepth aug
    cfg.INPUT.CROP.WITH_PAD = True
    cfg.INPUT.CROP.RESCALE = (0.8, 1.2)

    # DEPTH_FORMER
    cfg.MODEL.DEPTH_FORMER = CN()
    cfg.MODEL.DEPTH_FORMER.DEPTH_DIM = 256
    cfg.MODEL.DEPTH_FORMER.DEPTH_MAX = 88.
    cfg.MODEL.DEPTH_FORMER.SILOG_WEIGHT = 1.0
    cfg.MODEL.DEPTH_FORMER.REL_SQR_WEIGHT = 1.0
    cfg.MODEL.DEPTH_FORMER.REL_ABS_WEIGHT = 1.0
    cfg.MODEL.DEPTH_FORMER.TRANSFORMER_DECODER_NAME = "VideoMultiScaleDepthTransformerDecoder_frame"

    # UNIFIED_FORMER
    cfg.MODEL.UNIFIED_FORMER = CN()

    # loss
    cfg.MODEL.UNIFIED_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.UNIFIED_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.UNIFIED_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.UNIFIED_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.UNIFIED_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.UNIFIED_FORMER.NHEADS = 8
    cfg.MODEL.UNIFIED_FORMER.DROPOUT = 0.1
    cfg.MODEL.UNIFIED_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.UNIFIED_FORMER.ENC_LAYERS = 0
    cfg.MODEL.UNIFIED_FORMER.DEC_LAYERS = 6
    cfg.MODEL.UNIFIED_FORMER.PRE_NORM = False

    cfg.MODEL.UNIFIED_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.UNIFIED_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.UNIFIED_FORMER.ENFORCE_INPUT_PROJ = False

    cfg.MODEL.UNIFIED_FORMER.DEPTH_DIM = 256
    cfg.MODEL.UNIFIED_FORMER.DEPTH_MAX = 80.
    cfg.MODEL.UNIFIED_FORMER.SILOG_WEIGHT = 1.0
    cfg.MODEL.UNIFIED_FORMER.REL_SQR_WEIGHT = 1.0
    cfg.MODEL.UNIFIED_FORMER.REL_ABS_WEIGHT = 1.0

    cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_DECODER_NAME = "VideoMultiScaleMaskedTransformerDecoder_frame_unified_decoder"
    cfg.MODEL.UNIFIED_FORMER.SIZE_DIVISIBILITY = 32

    # UNIFIED_FORMER inference config
    cfg.MODEL.UNIFIED_FORMER.TEST = CN()
    cfg.MODEL.UNIFIED_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.UNIFIED_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.UNIFIED_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.UNIFIED_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.UNIFIED_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.UNIFIED_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.UNIFIED_FORMER.TEST.WINDOW_INFERENCE = False

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.UNIFIED_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.UNIFIED_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.UNIFIED_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # MATCHER
    cfg.MODEL.UNIFIED_FORMER.MATCHER = "video_depth_matcher"