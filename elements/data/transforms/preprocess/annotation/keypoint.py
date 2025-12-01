import copy

import numpy as np

from elements.data.datatypes.annotations.keypoint import Keypoint
from elements.data.datatypes.samplecontainer import SampleContainer, AnnotationDict


class KeypointsToNumpySC:
    """
    Convert Keypoint annotations to a np.ndarray

    'format' controls the contents of the output array:
            'x': x of the keypoint
            'y': y of the keypoint
            'v': visibility of the keypoint
            'c': class ID of the keypoint
            '0': zero
    """
    def __init__(self, format: str = "xyc", relative: bool = False, dtype=np.int32):
        self._format = format
        self._relative = relative
        self._dtype = dtype
        if any([char not in "xyvc0" for char in format]):
            raise ValueError("Format contains invalid characters")

    def __call__(self, sample: SampleContainer) -> SampleContainer:
        annotation_dicts = sample.annotations
        if isinstance(annotation_dicts, AnnotationDict):
            annotation_dicts = [annotation_dicts]

        for adict in annotation_dicts:
            if not adict.has(Keypoint):
                continue

            keypoints = adict.get(Keypoint)
            if not isinstance(keypoints, list):
                keypoints = [keypoints]

            result = np.empty((len(keypoints), len(self._format)), dtype=self._dtype)
            for k_id, keypoint in enumerate(keypoints):
                assert isinstance(keypoint, Keypoint)
                if keypoint.relative != self._relative:  # convert to desired relative mode if required
                    keypoint = copy.deepcopy(keypoint)
                    keypoint.relative = self._relative
                result[k_id] = keypoint.get_formatted_np(self._format, dtype=self._dtype)

            # update the Keypoint annotations
            adict.set(Keypoint, result)

        return sample
