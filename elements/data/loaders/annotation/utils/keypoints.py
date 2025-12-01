from elements.data.datatypes.annotations.keypoint import Keypoint
import xml.etree.ElementTree as ET


def get_keypoints_cvat(xml_str: str) -> list[Keypoint]:
    """
    Get key-points from CVAT for images 1.1 image xml element.
    Annotated keypoints are sorted by class name, that way they are always returned in the same order.

    :return: list of Keypoint objects.
    """
    tree = ET.fromstring(xml_str)
    keypoints: list[Keypoint] = []
    if tree is not None:
        for point in tree.iter("points"):
            coords = point.attrib["points"].split(",")
            x_coord, y_coord = int(round(float(coords[0], ), 0)), int(round(float(coords[1], ), 0))

            keypoint = Keypoint()
            keypoint.set_x_y_v(x=x_coord, y=y_coord, v=2)
            keypoints.append(keypoint)
    return keypoints
