import numpy as np

from elements.data.datatypes.annotation import AnnotationBase


class Keypoint(AnnotationBase):
    """
    Representation of a keypoint
    :param class_id: ID of the class for this keypoint
    """
    def __init__(self, x=0, y=0, class_id: int = -1, class_name: str = ""):
        super().__init__()
        self._class_id = class_id
        self._class_name = class_name
        self._x = x
        self._y = y
        self._v = 1

    @property
    def class_name(self):
        return self._class_name

    @class_name.setter
    def class_name(self, class_name: str):
        self._class_name = class_name

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def v(self):
        return self._v

    def get_class_id(self) -> int:
        """
        Get id of class associated with keypoint

        :return: class id
        """
        return self._class_id

    def set_class_id(self, class_id: int):
        """
        Set class-id associated with keypoint
        @param class_id: Class id
        """
        self._class_id = class_id

    def set_x_y_v(self, x: int, y: int, v: int):
        """
        Setter for values corresponding to keypoint:
        @param x: x coordinate
        @param y: y coordinate
        @param v: visibility of keypoint
        """
        x = int(x)
        y = int(y)
        v = int(v)

        self._x = x
        self._y = y
        self._v = v

    def get_x_y_v(self):
        """
        Get x-coordinate, y-coordinate and visibility value for the keypoint
        @return: tuple with x, y and v values
        """
        x = self._x
        y = self._y
        v = self._v
        return x, y, v

    def resize(self, scale_x: float, scale_y: float) -> 'Keypoint':
        self.x = self._x * scale_x
        self.y = self._y * scale_y
        return self

    def to_array(self) -> np.ndarray:
        x, y, v = self.get_x_y_v()
        return np.array([x, y, v, self._class_id], dtype="int")


def resize_keypoints(keypoints: list[Keypoint], scale_x: float, scale_y: float) -> list[Keypoint]:
    """
    Resize a list of keypoints by the given scale factors.

    :param keypoints: List of Keypoint objects
    :param scale_x: Horizontal scaling factor
    :param scale_y: Vertical scaling factor
    :return: List of resized Keypoint objects
    """
    return [kp.resize(scale_x, scale_y) for kp in keypoints]
