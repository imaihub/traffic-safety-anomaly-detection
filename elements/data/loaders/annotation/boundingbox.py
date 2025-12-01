import os
from typing import Callable

from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.keypoint import Keypoint
from elements.data.datatypes.annotations.mask import InstanceMask
from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.annotation.utils.bboxes import json_to_boxes, xml_to_boxes, yolo_to_boxes, get_bboxes_cvat
from elements.data.loaders.annotation.utils.supervisely import load_svly_as_boxes

loaders: dict[str, Callable[[str], list[BoundingBox]]] = {'.xml': get_bboxes_cvat, '.json': json_to_boxes, '.txt': yolo_to_boxes}


class LoadBoundingBoxes:
    def __call__(self, path: str) -> list[BoundingBox]:
        bboxes: list[BoundingBox] = []
        for ext, func in loaders.items():
            fn: str = os.path.splitext(path)[0] + ext
            if os.path.isfile(fn):
                with open(fn, "r") as f:
                    bboxes = loaders[ext]("".join(f.readlines()))
            return bboxes
        return bboxes


class LoadBoundingBoxesSC:
    """A class that loads bounding boxes from various file formats and stores them in a `SampleContainer` object."""

    loaders: dict[str, Callable[[str], list[BoundingBox]]] = {'.xml': xml_to_boxes, '.json': json_to_boxes, '.txt': yolo_to_boxes}

    def __init__(self, img_width: int = None, img_height: int = None):
        self._img_width = img_width
        self._img_height = img_height

    def get_types(self) -> list[str]:
        """Returns the file types supported by this object.

        :returns: a list of strings representing the file extensions (including the leading period) supported by this object.
        :rtype: list of str
        """
        return list(self.loaders.keys())

    def __call__(self, sample: SampleContainer) -> SampleContainer:
        """Loads bounding boxes for a given `SampleContainer`.

        :param sample: a `SampleContainer` object.
        :type sample: SampleContainer
        :returns: the modified `SampleContainer` object, with bounding boxes loaded and stored in the `annotations` attribute.
        :rtype: SampleContainer
        """
        bboxes: list[BoundingBox] = []
        for ext, func in self.loaders.items():
            fn: str = os.path.splitext(sample.annotations_fpaths.get(BoundingBox))[0] + ext
            if os.path.isfile(fn):
                with open(fn, "r") as f:
                    bboxes = self.loaders[ext]("".join(f.readlines()))
            sample.annotations.set(BoundingBox, bboxes)
        return sample


class LoadInstanceMasksAsBoundingBoxesSC:
    def __call__(self, sample: SampleContainer) -> SampleContainer:
        inst_mask_fpath = sample.annotations_fpaths.get(InstanceMask, None)

        ext = os.path.splitext(inst_mask_fpath)[1][1:]
        if ext == "json":
            bboxes = load_svly_as_boxes(inst_mask_fpath)
            sample.annotations.set(BoundingBox, bboxes)
        else:
            raise ValueError(f"Unsupported instance mask format: {ext}")
        return sample
