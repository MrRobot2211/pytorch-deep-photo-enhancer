from __future__ import absolute_import, division

from types import LambdaType
import math
import random
import warnings

import cv2
import numpy as np

from albumentations.augmentations import functional as F
import albumentations
from albumentations.core.transforms_interface import to_tuple, DualTransform, ImageOnlyTransform, NoOp
from albumentations.augmentations.bbox_utils import normalize_bbox, denormalize_bbox


class PadDifferentlyIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        p (float): probability of applying the transform. Default: 1.0.
        value (list of ints [r, g, b]): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int): padding value for mask if border_mode is cv2.BORDER_CONSTANT.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32

    """

    def __init__(self, min_height=1024, min_width=1024, border_mode=cv2.BORDER_REFLECT_101,mask_border_mode = cv2.BORDER_CONSTANT ,
                 value=None, mask_value=None, always_apply=False, p=1.0):
        super(PadDifferentlyIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.mask_border_mode = mask_border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadDifferentlyIfNeeded, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']


        if rows < self.min_height:
            h_pad_top = int((self.min_height - rows) / 2.0)
            h_pad_bottom = self.min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_left = int((self.min_width - cols) / 2.0)
            w_pad_right = self.min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

       
        params.update({'pad_top': h_pad_top,
                       'pad_bottom': h_pad_bottom,
                       'pad_left': w_pad_left,
                       'pad_right': w_pad_right
                    })
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.mask_border_mode, value=self.mask_value)

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, a, s = keypoint
        return [x + pad_left, y + pad_top, a, s]

    def get_transform_init_args_names(self):
        return ('min_height', 'min_width', 'border_mode', 'value', 'mask_value')


