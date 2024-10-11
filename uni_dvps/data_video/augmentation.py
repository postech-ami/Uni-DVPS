# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import numpy as np
import logging
import sys
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from PIL import Image

from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from typing import Tuple
from fvcore.transforms.transform import (
    CropTransform,
    PadTransform,
    TransformList,
)

class FixedSizeCenterCrop(T.Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a center crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the around of the image to the crop size.
    """

    def __init__(self, crop_size: Tuple[int], pad_value: float = 128.0, with_pad=True):
        """
        Args:
            crop_size: target image (height, width).
            pad_value: the padding value.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> TransformList:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, 0.5)#np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        crop_transform = CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )
        if not self.with_pad:
            return TransformList([crop_transform, ])

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        pad_size_0 = pad_size // 2
        pad_size_1 = pad_size - pad_size_0
        original_size = np.minimum(input_size, output_size)
        pad_transform = PadTransform(
            pad_size_0[1], pad_size_0[0], pad_size_1[1], pad_size_1[0], original_size[1], original_size[0], self.pad_value
        )

        return TransformList([crop_transform, pad_transform])

class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            if self.size == 0:
                return NoOpTransform()

            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)

class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()

def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    aug_list = []
    print("aug_list: ", aug_list)
    return aug_list

def build_semkitti_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    aug_list = []
    print("aug_list: ", aug_list)
    return aug_list
