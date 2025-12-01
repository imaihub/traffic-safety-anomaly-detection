from typing import List

import numpy as np

from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.keypoint import Keypoint


def decode_yolo_boxes(r, threshold: float = 0.15) -> list[BoundingBox]:
    boxes: List[BoundingBox] = []
    xyxy = getattr(r.boxes, "xyxy", None)
    boxes_np = xyxy.detach().cpu().numpy()

    # classes & confs (handle torch tensors or numpy)
    cls_attr = getattr(r.boxes, "cls", None)
    conf_attr = getattr(r.boxes, "conf", None)

    if cls_attr is not None:
        classes_np = cls_attr.detach().cpu().numpy() if hasattr(cls_attr, "detach") else np.asarray(cls_attr)
    else:
        classes_np = np.zeros((boxes_np.shape[0], ), dtype=int)

    if conf_attr is not None:
        confs_np = conf_attr.detach().cpu().numpy() if hasattr(conf_attr, "detach") else np.asarray(conf_attr)
    else:
        confs_np = np.full((boxes_np.shape[0], ), -1.0)

    for i, (x1, y1, x2, y2) in enumerate(boxes_np):
        if float(confs_np[i]) > threshold:
            bb = BoundingBox(class_id=int(classes_np[i]) if classes_np is not None else -1)
            bb.class_name = r.names[bb.get_class_id()]
            bb.set_minmax_xy(float(x1), float(y1), float(x2), float(y2))
            if confs_np is not None:
                bb.set_confidence(float(confs_np[i]))
            boxes.append(bb)
    return boxes


def decode_yolo_keypoints(r) -> list[Keypoint]:
    keypoints: list[Keypoint] = []
    for kp in r.keypoints:
        body_keypoints = kp.xy.cpu().numpy().flatten()
        if body_keypoints.size == 0:
            continue
        it_len = len(body_keypoints) // 2
        for it in range(it_len):
            keypoints.append(Keypoint(x=body_keypoints[it * 2], y=body_keypoints[it * 2 + 1], class_id=0))

    return keypoints
