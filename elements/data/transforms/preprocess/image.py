from typing import Optional

import cv2
import numpy as np
import skimage
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img)


class ToFloat:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32)


class ConvertRGB:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class HWC2CHW:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (2, 0, 1))


class AddBatchDim:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img.unsqueeze(0)


class ToHalf:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img.half()


class ToCUDA:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img.cuda()


class Resize:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(img, torch.Tensor):
            # Torch expects (N, C, H, W)
            if img.ndim == 3:  # (C, H, W)
                img = img.unsqueeze(0)
            elif img.ndim != 4:
                raise ValueError(f"Unexpected tensor shape {img.shape}, expected (C,H,W) or (N,C,H,W)")
            resized = F.interpolate(img, size=(self.height, self.width), mode="bilinear", align_corners=False)
            if resized.shape[0] == 1:  # remove batch dim
                resized = resized.squeeze(0)
            return resized
        elif isinstance(img, np.ndarray):
            if img.shape[2] == 2:
                return skimage.transform.resize(img, (self.height, self.width))
            elif img.shape[2] == 3:
                return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            raise NotImplementedError("Image to resize is of shape with number of dimensions other than 2 or 3")
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")


def resize_image(img: torch.Tensor | np.ndarray, width: int, height: int) -> torch.Tensor | np.ndarray:
    """
    Ad-hoc resizing function, similar behavior to Resize class.
    """
    if isinstance(img, torch.Tensor):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        elif img.ndim != 4:
            raise ValueError(f"Unexpected tensor shape {img.shape}, expected (C,H,W) or (N,C,H,W)")
        resized = F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)
        if resized.shape[0] == 1:
            resized = resized.squeeze(0)
        return resized
    elif isinstance(img, np.ndarray):
        if img.shape[2] == 2:
            return skimage.transform.resize(img, (height, width))
        elif img.shape[2] == 3:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        raise NotImplementedError("Image to resize is of shape with number of channels other than 2 or 3")
    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")
