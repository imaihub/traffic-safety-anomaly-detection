import numpy as np

from elements.data.datatypes.annotations.boundingbox import BoundingBox


def yxyx2boxes(boxes_yx: list[np.ndarray]) -> list[BoundingBox]:
    boxes = []
    for box_yx in boxes_yx:
        box = BoundingBox(class_id=int(box_yx[4]))
        box.confidence = float(box_yx[5])
        box.set_minmax_xy(xmin=float(box_yx[1]), xmax=float(box_yx[3]), ymin=float(box_yx[0]), ymax=float(box_yx[2]))
        boxes.append(box)
    return boxes
