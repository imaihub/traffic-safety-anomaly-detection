import cv2
import numpy as np
from typing import Optional, Union
from collections import defaultdict

from elements.data.datatypes.annotations.boundingbox import BoundingBox


def mask_to_single_channel(mask: np.ndarray):
    # if mask has three dimensions convert to two dimensions
    if len(mask.shape) == 3:
        # get unique values (e.g. RGB values) (background is removed).
        unique_vals = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)[1:]

        # convert unique values to instance ids 1 .. len(unique_vals)
        empty_mask = np.zeros(mask.shape[:-1], dtype=np.uint8)
        for i, unique_c in enumerate(unique_vals):
            empty_mask[np.all(mask == unique_c, axis=-1)] = i + 1
        mask = empty_mask
    return mask


def instance_mask_to_boxes(mask: np.ndarray, class_mask: np.ndarray = None, class_ids: list[int] = None, class_names: list[str] = None, minimal_object_area: Union[int, float] = 1) -> list[BoundingBox]:
    """
    Extract boxes from mask with shape [h,w,c] or [h,w]

    :param mask: the mask with unique instance ids
    :param class_ids: a list of class_ids corresponding to the instance_ids
    :param class_names: a list of class_names corresponding to the class_ids
    :param minimal_object_area: minimal area of instance to convert to bounding box
    :return: a list of BoundingBox
    """
    if len(mask.shape) == 3:
        mask = mask.mean(axis=2)
        # mask = mask[:, :, 2]

    if isinstance(minimal_object_area, float):
        minimal_object_area = int((mask.shape[1] * mask.shape[0]) * minimal_object_area)

    result = []
    instance_ids = [id for id in np.unique(mask) if id != 0]
    if class_ids is None:
        class_ids = [1] * len(instance_ids)

    if class_names is None:
        class_names = ["Object"] * len(instance_ids)

    for class_id, class_name, instance_id in zip(class_ids, class_names, instance_ids):
        instance_mask = (mask == instance_id)
        if class_mask is not None:
            class_id = class_mask[instance_mask][0].item()
        yy, xx = np.where(instance_mask)
        y1, y2, x1, x2 = np.min(yy), np.max(yy) + 1, np.min(xx), np.max(xx) + 1
        b = BoundingBox(class_id, class_name=class_name)
        b.set_minmax_yx(y1, x1, y2, x2)
        if b.area > minimal_object_area:
            # add bounding box if minimal object area is met
            result.append(b)
        else:
            # if minimal object area is not met, remove instance itself as well.
            mask[mask == instance_id] = 0
    return result


def class_names_to_unique_ids(class_names: list[str]) -> list[int]:
    d_dict = defaultdict(lambda: len(d_dict))
    return [d_dict[n] for n in class_names]


def ids_to_one_hot(x):
    obj_ids = np.unique(x)
    obj_ids = obj_ids[1:]
    masks = x == obj_ids[:, None, None]
    masks = np.moveaxis(masks, 0, 2)
    return masks


def class_ids_to_one_hot(class_mask, num_classes, class_indices):
    h, w = class_mask.shape
    masks = np.zeros([h, w, num_classes], dtype=np.bool_)
    class_ids = np.unique(class_mask).astype(np.int32)
    for class_id in class_ids:
        class_idx = class_indices[class_id]
        if class_idx > num_classes:
            raise RuntimeError(f"The class index {class_idx} for class id {class_id} is out of range.")
        masks[:, :, class_idx] = class_mask == class_id
    return masks.astype('uint8')


def instance_ids_to_one_hot(instance_mask):
    h, w = instance_mask.shape
    mx = np.max(instance_mask).astype(np.int32) + 1
    masks = np.zeros([mx, h, w], dtype=np.bool_)
    instance_ids = np.unique(instance_mask).astype(np.int32)
    for instance_id in instance_ids:
        masks[instance_id, :, :] = instance_mask == instance_id
    return masks.astype('uint8')


def class_mask_to_bounding_box(mask, class_ids: Optional[list[int]] = None, minimal_object_area: Union[int, float] = 1):
    boxes = []
    if isinstance(minimal_object_area, float):
        minimal_object_area = int((mask.shape[1] * mask.shape[0]) * minimal_object_area)

    # automatically determine class ids
    if class_ids is None:
        class_ids = [x for x in np.unique(mask) if x != 0]

    for lbl in class_ids:
        lbl_mask = (mask == lbl).astype(np.uint8)
        contours, hierarchy = cv2.findContours(lbl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = BoundingBox(lbl)
            bbox.set_ltwh(x, y, w, h)
            if bbox.area > minimal_object_area:
                boxes.append(bbox)

    return boxes


def bounding_box_to_class_mask(boxes, height, width, center):
    mask = np.zeros((height, width), dtype='uint8')
    for box in boxes:
        xmin, ymin, xmax, ymax, class_id = box
        if center:
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            mask[int(yc), int(xc)] = int(class_id)
        else:
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = class_id
    return mask.astype('uint8')
