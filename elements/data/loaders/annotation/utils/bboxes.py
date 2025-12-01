import json

import xml.etree.ElementTree as ET

from elements.data.datatypes.annotations.boundingbox import BoundingBox


def get_bboxes_cvat(xml_str: str) -> list[BoundingBox]:
    """
    Get bounding boxes from CVAT for images 1.1 xml image xml element.

    :return: list of BoundingBox objects.
    """
    tree = ET.fromstring(xml_str)
    bboxes: list[BoundingBox] = []
    if tree is not None:
        for box in tree.iter("box"):
            xmin, ymin, xmax, ymax = float(box.attrib["xtl"]), float(box.attrib["ytl"]), \
                float(box.attrib["xbr"]), float(box.attrib["ybr"])
            class_id = -1

            bbox = BoundingBox()
            bbox.set_minmax_xy(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            bbox.set_class_id(class_id)
            bbox.class_name = box.attrib["label"]
            bboxes.append(bbox)
    return bboxes


def xml_to_boxes(xml_str: str) -> list[BoundingBox]:
    tree = ET.fromstring(xml_str)
    bbs = []
    if tree is not None:
        for o in tree.iter("object"):
            class_name = o.find("name").text
            b = o.find("bndbox")
            y1 = int(float(b.find("ymin").text))
            x1 = int(float(b.find("xmin").text))
            y2 = int(float(b.find("ymax").text))
            x2 = int(float(b.find("xmax").text))
            bbox = BoundingBox()
            bbox.set_minmax_yx(ymin=y1, xmin=x1, ymax=y2, xmax=x2)
            bbox.class_name = class_name
            bbs.append(bbox)
    return bbs


def json_to_boxes(json_str: str) -> list[BoundingBox]:
    bbs = []
    content = json.loads(json_str)
    for shape in content['shapes']:
        p = shape['points']
        if shape['shape_type'] == 'circle':
            cx, cy, ex, ey = int(p[0][0]), int(p[0][1]), int(p[1][0]), int(p[1][1])
            r = ((ex - cx)**2 + (ey - cy)**2)**0.5
            x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        elif shape['shape_type'] == 'rectangle':
            x1 = shape['points'][0][0]
            y1 = shape['points'][0][1]
            x2 = shape['points'][1][0]
            y2 = shape['points'][1][1]
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            x1, y1, x2, y2 = xmin, ymin, xmax, ymax
        else:
            raise RuntimeError(f"Unsupported shape type: {shape['shape_type']}")

        class_name = shape['label']
        bbox = BoundingBox()
        bbox.set_minmax_yx(ymin=y1, xmin=x1, ymax=y2, xmax=x2)
        bbox.class_name = class_name
        bbs.append(bbox)
    return bbs


def yolo_to_boxes(txt_str: str) -> list[BoundingBox]:
    bbs = []
    lines = txt_str.strip().split("\n")
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            bbox = BoundingBox(class_id=class_id)
            bbox.set_centre_xy(x_center, y_center, width, height, relative=True)
            bbs.append(bbox)
    return bbs
