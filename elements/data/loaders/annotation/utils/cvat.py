import re
import xml.etree.ElementTree as ET

from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.keypoint import Keypoint


def get_classes_cvat(cvat_xml_root: ET.Element) -> dict[str, int]:
    """
    Get classes and corresponding class ids from CVAT for images 1.1 xml root.

    :param cvat_xml_root: CVAT xml root, can be loaded with ET.parse('example.xml').getroot()

    :return: dictionary with classname as key and class id as value for both keypoints and bounding box classes
    """
    labels = cvat_xml_root.find('meta').find('job').find('labels')
    keypoint_labels = sorted([label.find('name').text for label in labels if label.find('type').text == "points"])
    bbox_labels = sorted([label.find('name').text for label in labels if label.find('type').text == "rectangle"])
    return {name: i for i, name in enumerate(keypoint_labels, start=1)}, {name: i for i, name in enumerate(bbox_labels, start=1)}


def get_bboxes_cvat(image_el: ET.Element, class_names_ids: dict[str, int]) -> list[BoundingBox]:
    """
    Get bounding boxes from CVAT for images 1.1 xml image xml element.

    :param image_el: image xml element, for each: ET.parse('example.xml').getroot().findall('image')
    :param class_names_ids: dictionary with classnames as keys and corresponding class ids as values.

    :return: list of BoundingBox objects.
    """
    bboxes: list[BoundingBox] = []
    for box in image_el.findall('box'):
        xmin, ymin, xmax, ymax = float(box.attrib["xtl"]), float(box.attrib["ytl"]), \
            float(box.attrib["xbr"]), float(box.attrib["ybr"])
        class_id = class_names_ids[box.attrib["label"]]

        bbox = BoundingBox()
        bbox.set_minmax_xy(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        bbox.set_class_id(class_id)
        bbox.class_name = box.attrib["label"]
        bboxes.append(bbox)
    return bboxes


def get_keypoints_cvat(image_el: ET.Element, class_names_ids: dict[str, int]) -> list[Keypoint]:
    """
    Get key-points from CVAT for images 1.1 image xml element.
    Annotated keypoints are sorted by class name, that way they are always returned in the same order.

    :param image_el: image xml element, for each: ET.parse('example.xml').getroot().findall('image')
    :param class_names_ids: dictionary with classnames as keys and corresponding class ids as values.

    :return: list of Keypoint objects.
    """
    keypoints: list[Keypoint] = []
    sorted_points = sorted(image_el, key=lambda child: (child.attrib['label'], child.get('name')))
    for point in sorted_points:
        if point.tag != "points":
            continue
        coords = [int(float(x)) for x in re.split(',|;', point.attrib["points"])]
        x_coord, y_coord = coords[0], coords[1]
        class_id = class_names_ids[point.attrib["label"]]

        keypoint = Keypoint()
        keypoint.set_x_y_v(x=x_coord, y=y_coord, v=2)

        keypoint.set_class_id(class_id)
        keypoint.class_name = point.attrib["label"]

        keypoints.append(keypoint)
    return keypoints
