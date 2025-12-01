import numpy as np
from skimage.morphology import closing, opening

from elements.data.datatypes.annotations.mask import Mask
from elements.data.datatypes.samplecontainer import SampleContainer


class ClosingMasksSC:
    def __init__(self, mask_type: Mask, kernel: np.ndarray = np.ones((3, 3))):
        self._mask_type = mask_type
        self._kernel = kernel

    def __call__(self, sample: SampleContainer):
        mask = sample.annotations.get(self._mask_type).mask
        mask = closing(mask, self._kernel)
        sample.annotations.set(self._mask_type, self._mask_type(mask))
        return sample


class OpeningMasksSC:
    def __init__(self, mask_type: Mask, kernel: np.ndarray = np.ones((3, 3))):
        self._mask_type = mask_type
        self._kernel = kernel

    def __call__(self, sample: SampleContainer):
        mask = sample.annotations.get(self._mask_type).mask
        mask = opening(mask, self._kernel)
        sample.annotations.set(self._mask_type, self._mask_type(mask))

        return sample
