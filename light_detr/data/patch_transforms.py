import random

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms as T

import cv_lib.augmentation.aug as aug


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x: Image.Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_det_transforms(mean, std):
    scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]
    trans = aug.Compose(
        aug.RandomResize(scales, max_size=600),
        aug.ToTensor(),
        aug.Normalize(mean, std)
    )
    return trans


def get_query_transforms(mean, std):
    trans = T.Compose([
        T.Resize((128, 128)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    return trans
