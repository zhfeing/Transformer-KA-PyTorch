# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
pre-training dataset which implements random query patch detection.
"""
import random
from typing import Callable, Tuple, Union, Dict, Any, List

from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as tf
import torchvision.ops.boxes as box_ops

from light_detr.utils import ErrorBBOX


__all__ = [
    "get_random_patch_from_img",
    "PatchDataset",
]


def get_random_patch_from_img(
    img: Union[Tensor, Image.Image],
    min_pixel: int = 8
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        min_pixel: min pixels of the query patch
    """
    w, h = tf._get_image_size(img)
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    assert (max_h > min_h) and (max_w > min_w), f"max_h {max_h} min_h {min_h} max_w {max_w} min_w {min_w}"
    patch_size = random.randint(min_h, max_h), random.randint(min_w, max_w)
    i, j, h, w = T.RandomCrop.get_params(img, patch_size)
    patch = tf.crop(img, i, j, h, w)
    bbox = torch.tensor([j, i, w, h], dtype=torch.float)
    return patch, bbox


class PatchDataset(Dataset):
    """
    PatchDataset is a dataset class which implements random query patch from an unlabeled dataset.
    It randomly crops patches as queries from the given image with the corresponding bounding box.

    Args:
        unlabeled_dataset: the output must be Image.Image
    """
    def __init__(
        self,
        unlabeled_dataset: Dataset,
        detection_transform,
        query_transform: Callable[[Union[Tensor, Image.Image]], Tensor],
        num_patches: int = 10
    ):
        super().__init__()
        self.unlabeled_dataset = unlabeled_dataset
        self.detection_transform = detection_transform
        self.query_transform = query_transform
        self.num_patches = num_patches

    def __len__(self):
        return len(self.unlabeled_dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, List[Tensor], Dict[str, Any]]:
        img, *_ = self.unlabeled_dataset[index]
        assert isinstance(img, Image.Image), "the output out unlabeled dataset must be Image.Image"
        img_w, img_h = tf._get_image_size(img)
        # the format of the dataset is same with DetectionDataset.
        target = {
            "image_id": index,
            "orig_size": (img_h, img_w),
            "size": (img_h, img_w)
        }
        boxes = []
        patches = []
        for _ in range(self.num_patches):
            patch, bbox = get_random_patch_from_img(img)
            boxes.append(bbox)
            patch = self.query_transform(patch)
            patches.append(patch)
        # 0: background, 1: no-obj 2: obj
        target["labels"] = torch.full((self.num_patches,), fill_value=2, dtype=torch.long)
        # torch.full(self.num_patches, fill_value=2, dtype=torch.long)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, [0, 2]] /= img_w
        boxes[:, [1, 3]] /= img_h
        boxes = box_ops.box_convert(boxes, "xywh", "cxcywh")
        if not torch.greater_equal(boxes[:, 2:], 0).all():
            raise ErrorBBOX(boxes, f"index: {index}")
        target["boxes"] = boxes

        img, target = self.detection_transform(img, target)
        patches = torch.stack(patches)
        return img, patches, target

