from typing import Optional, Union

import torch
import numpy as np

from ..annotation import AnnotationBase


class Mask(AnnotationBase):
    """
    Representation of 2D mask
    """

    def __init__(self, mask: Optional[Union[np.ndarray, torch.Tensor]]):
        super().__init__()
        self._mask: Optional[Union[np.ndarray, torch.Tensor]] = mask

    @property
    def mask(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        return self._mask

    @mask.setter
    def mask(self, mask: Union[np.ndarray, torch.Tensor]):
        self._mask = mask

    def to_array(self) -> Union[np.ndarray, torch.Tensor]:
        if self._mask is not None:
            return self._mask
        else:
            return np.empty((0, 0), dtype=np.int32)
        pass


class SegmentationMask(Mask):
    """
    Segmentation mask
    """

    def __init__(self, mask: Optional[np.ndarray]):
        super().__init__(mask)


class InstanceMask(Mask):
    """
    Instance mask
    """

    def __init__(self, mask: Optional[np.ndarray]):
        super().__init__(mask)


class ClassMask(Mask):
    """
    Class mask
    """

    def __init__(self, mask: Optional[np.ndarray]):
        super().__init__(mask)
