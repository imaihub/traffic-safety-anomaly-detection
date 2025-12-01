import base64
import json
import zlib
import cv2
from skimage.draw import polygon
import numpy as np

from elements.data.datatypes.annotations.boundingbox import BoundingBox


def load_svly_as_class_ids(filename: str) -> tuple[np.ndarray, dict]:
    """
    Load the JSON format of supervisely.
    Polygons from this format are written to the mask image.

    :param filename: input filename
    :return: a mono image where each pixel represents the class id
    """

    with open(filename, "r") as f:
        lines = "".join(f.readlines())
        content = json.loads(lines)
        height, width = int(content["size"]["height"]), int(content["size"]["width"])
        class_ids = np.zeros([height, width])
        class_ids_names = {}
        for i, obj in enumerate(content["objects"]):
            class_id = int(obj["classId"])
            class_ids_names[class_id] = obj["classTitle"]

            if obj["geometryType"] == "polygon":
                poly = np.array(obj["points"]["exterior"])
                rr, cc = polygon(poly[:, 1], poly[:, 0], class_ids.shape)
                class_ids[rr, cc] = class_id
            elif obj['geometryType'] == 'bitmap':
                # Decode bitmap from supervisely
                z = zlib.decompress(base64.b64decode(obj['bitmap']['data']))
                n = np.fromstring(z, np.uint8)
                mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
                # Bounding box of object
                x1, y1 = obj['bitmap']['origin']
                x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
                # Overlap object with current mask
                class_ids[y1:y2, x1:x2] = np.where(mask > 0, class_id, class_ids[y1:y2, x1:x2])
            else:
                print(f"Warning: loading of type '{obj['geometryType']}' not implemented.")
        return class_ids, class_ids_names


def load_svly_as_boxes(filename: str) -> list[BoundingBox]:
    """
    Load the JSON format of supervisely.
    Polygons from this format are converted to BoundingBox

    :param filename: input filename
    :return: a List of BoundingBox
    """
    bboxes = []
    with open(filename, "r") as f:
        lines = "".join(f.readlines())
        content = json.loads(lines)
        height, width = int(content["size"]["height"]), int(content["size"]["width"])
        class_ids = np.zeros([height, width])
        for i, obj in enumerate(content["objects"]):
            if obj["geometryType"] == "polygon":
                poly = np.array(obj["points"]["exterior"])
                class_id = int(obj["classId"])
                class_name = obj["classTitle"]
                y1, y2, x1, x2 = min(poly[:, 1]), max(poly[:, 1]), min(poly[:, 0]), max(poly[:, 0])
                bbox = BoundingBox()
                bbox.set_minmax_yx(ymin=y1, xmin=x1, ymax=y2, xmax=x2)
                bbox.set_class_id(class_id)
                bbox.class_name = class_name
                bboxes.append(bbox)
            else:
                print(f"Warning: loading of type '{obj['geometryType']}' not implemented.")
    return bboxes


def load_svly_as_inst_ids(filename: str) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Load the JSON format of supervisely.
    Polygons from this format are written to the mask image.

    :param filename: input filename
    :return: a mono image where each pixel represents the instance id
    """
    with open(filename, "r") as f:
        class_ids = []
        class_names = []
        lines = "".join(f.readlines())
        content = json.loads(lines)
        height, width = int(content["size"]["height"]), int(content["size"]["width"])
        instance_ids = np.zeros([height, width], dtype=np.uint16)
        for i, obj in enumerate(content["objects"]):
            instance_id = i + 1
            if obj["geometryType"] == "polygon":
                class_id = int(obj["classId"])
                class_name = obj["classTitle"]
                poly = np.array(obj["points"]["exterior"])
                rr, cc = polygon(poly[:, 1], poly[:, 0], instance_ids.shape)
                instance_ids[rr, cc] = instance_id
                class_ids.append(class_id)
                class_names.append(class_name)
            elif obj['geometryType'] == 'bitmap':
                class_ids.append(int(obj["classId"]))
                class_names.append(obj["classTitle"])
                # Decode bitmap from supervisely
                z = zlib.decompress(base64.b64decode(obj['bitmap']['data']))
                n = np.fromstring(z, np.uint8)
                mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
                # Bounding box of object
                x1, y1 = obj['bitmap']['origin']
                x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
                # Overlap object with current mask
                instance_ids[y1:y2, x1:x2] = np.where(mask > 0, instance_id, instance_ids[y1:y2, x1:x2])
            else:
                print(f"Warning: loading of type '{obj['geometryType']}' not implemented.")
        return instance_ids, class_ids, class_names
