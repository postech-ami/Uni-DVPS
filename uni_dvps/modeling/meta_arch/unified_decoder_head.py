# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from mask2former.modeling.pixel_decoder.fpn import build_pixel_decoder
from ..transformer_decoder.unified_transformer_decoder import build_unified_transformer_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class UnifiedDecoderHead(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # version = local_metadata.get("version", None)
        # if version is None or version < 2:
        # Do not warn if train from scratch
        scratch = True
        logger = logging.getLogger(__name__)
        for k in list(state_dict.keys()):
            newk = k
            if "sem_seg_head" in k and k.startswith(prefix + "predictor"):
                newk = k.replace("predictor", "unified_decoder")
            if newk != k:
                state_dict[newk] = state_dict[k]
                del state_dict[k]
                scratch = False

        if not scratch:
            logger.warning(
                f"Weight format of {self.__class__.__name__} have changed! "
                "Please upgrade your models. Applying automatic conversion now ..."
            )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        unified_decoder: nn.Module,
        transformer_in_feature: str,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]
        self.num_classes = num_classes

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.unified_decoder = unified_decoder
        self.transformer_in_feature = transformer_in_feature

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for unidvps
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_IN_FEATURE,
            "unified_decoder": build_unified_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.unified_decoder(multi_scale_features, mask_features, mask)

        return predictions