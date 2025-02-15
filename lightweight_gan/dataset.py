from functools import partial
from pathlib import Path
from random import random

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lightweight_gan.utils import exists

EXTS = ["jpg", "jpeg", "png"]

# dataset
def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def identity(tensor):
    return tensor


def expand_greyscale(tensor, transparent):
    channels = tensor.shape[0]
    num_target_channels = 4 if transparent else 3

    if channels == num_target_channels:
        return tensor

    alpha = None
    if channels == 1:
        color = tensor.expand(3, -1, -1)
    elif channels == 2:
        color = tensor[:1].expand(3, -1, -1)
        alpha = tensor[1:]
    else:
        raise Exception(f"image with invalid number of channels given {channels}")

    if not exists(alpha) and transparent:
        alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

    return color if not transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


def random_apply(prob, fn1, fn2):
    """randomly apply one of two functions"""
    return fn1 if random() < prob else fn2


class ImageDataset(Dataset):
    def __init__(
        self, folder, image_size, transparent=False, greyscale=False, aug_prob=0.0
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        assert len(self.paths) > 0, f"No images were found in {folder} for training"

        if transparent:
            num_channels = 4
            pillow_mode = "RGBA"
            expand_fn = partial(expand_greyscale, transparent=transparent)
        elif greyscale:
            num_channels = 1
            pillow_mode = "L"
            expand_fn = identity
        else:
            num_channels = 3
            pillow_mode = "RGB"
            expand_fn = partial(expand_greyscale, transparent=False)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        self.transform = transforms.Compose(
            [
                transforms.Lambda(convert_image_fn),
                transforms.Lambda(partial(resize_to_minimum_size, image_size)),
                transforms.Resize(image_size),
                random_apply(
                    aug_prob,
                    transforms.CenterCrop(image_size),
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)
                    ),
                ),
                transforms.ToTensor(),
                transforms.Lambda(expand_fn),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
