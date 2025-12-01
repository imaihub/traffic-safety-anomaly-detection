import numpy as np

from typing import Union

from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.mask import ClassMask, InstanceMask
from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.transforms.utils.annotation import instance_mask_to_boxes


class InstanceMaskToBoundingBoxes:
    """
    Convert an instance mask to bounding boxes.
    """
    def __init__(self, minimal_object_area: Union[int, float] = 1):
        """

        :param minimal_object_area: minimal area in pixels the object should have to convert it to a bounding box.
        Can either be an integer (exact number of pixels) or a float which is the ratio of the bounding box combined with the total size of the mask.
        """
        self._minimal_object_area = minimal_object_area

    def __call__(self, sample: SampleContainer):
        class_ids = None
        class_names = None

        if ClassMask in sample.annotations.get_all_annotations().keys():
            instance_mask = sample.annotations.get(InstanceMask).mask
            class_mask = sample.annotations.get(ClassMask).mask

            # Convert instance mask to one hot
            mask_max = np.max(instance_mask)
            eye = np.eye(mask_max + 1)
            one_hot_mask = eye[instance_mask]
            labels = one_hot_mask * class_mask[:, :, None]
            class_ids = [int(cls_id) for cls_id in np.max(labels, axis=(0, 1)) if cls_id != 0]
            if len(class_ids) > 0 and sample.class_id is not None and sample.class_name is not None:
                id_to_name = {cls_id: cls_name for cls_id, cls_name in zip(sample.class_id, sample.class_name)}
                class_names = [id_to_name[cls_id] for cls_id in class_ids]
            else:
                class_ids = None
                class_names = None

        boxes = instance_mask_to_boxes(sample.annotations.get(InstanceMask).mask, class_mask=sample.annotations.get(ClassMask).mask if sample.annotations.get(ClassMask) is not None else None, minimal_object_area=self._minimal_object_area, class_ids=class_ids, class_names=class_names)
        sample.annotations.set(BoundingBox, boxes)
        return sample
