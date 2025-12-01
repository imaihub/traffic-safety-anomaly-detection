import cv2
import numpy as np
from typing import Optional, List, Literal

from elements.data.datatypes.annotation import AnnotationBase


class BoundingBox(AnnotationBase):
    """
    :param class_id: ID of the class for this bounding box
    """
    def __init__(self, class_id: int = -1, class_name: Optional[str] = None):
        super().__init__()
        self._class_id = class_id
        self._class_name = class_name
        self._confidence: Optional[float] = None
        self._xctr = 0.0
        self._yctr = 0.0
        self._width = 0.0
        self._height = 0.0

    def set_confidence(self, confidence: float):
        """
        set the confidence for this bounding box
        """
        self._confidence = confidence

    def get_confidence(self) -> float:
        """
        get the confidence for this bounding box
        """
        return self._confidence

    # ----------------------------------
    @property
    def y1(self):
        return self._yctr - self._height / 2

    @property
    def x1(self):
        return self._xctr - self._width / 2

    @property
    def y2(self):
        return self._yctr + self._height / 2

    @property
    def x2(self):
        return self._xctr + self._width / 2

    @property
    def area(self):
        return self._width * self._height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def y(self):
        return self._yctr

    @property
    def x(self):
        return self._xctr

    def set_minmax_xy(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """
        Set bounding box using min/max (left-top, right-bottom) coordinates
        :param xmin: lowest X coordinate (left)
        :param ymin: lowest Y coordinate (top)
        :param xmax: highest X coordinate (right)
        :param ymax: highest Y coordinate (bottom)
        :return:
        """
        # make sure everything is float
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        self._xctr = (xmin + xmax) / 2
        self._yctr = (ymin + ymax) / 2
        self._width = xmax - xmin
        self._height = ymax - ymin

    def set_minmax_yx(self, ymin: float, xmin: float, ymax: float, xmax: float):
        """
        Set bounding box using min/max (left-top, right-bottom) coordinates
        :param ymin: lowest Y coordinate (top)
        :param xmin: lowest X coordinate (left)
        :param ymax: highest Y coordinate (bottom)
        :param xmax: highest X coordinate (right)
        :return:
        """
        self.set_minmax_xy(xmin, ymin, xmax, ymax)

    def set_ltwh(self, left: float, top: float, width: float, height: float):
        """
        Set bounding box using left-top, width, and height.
        :param left: left X coordinate (xmin)
        :param top: top Y coordinate (ymin)
        :param width: width of box
        :param height: height of box
        :return:
        """
        # make sure everything is float
        left = float(left)
        top = float(top)
        width = float(width)
        height = float(height)

        self._xctr = left + width / 2
        self._yctr = top + height / 2
        self._width = width
        self._height = height

    def set_tlhw(self, top: float, left: float, height: float, width: float):
        """
        Set bounding box using top-left, height, and width.

        :param top: top Y coordinate (ymin)
        :param left: left X coordinate (xmin)
        :param height: height of box
        :param width: width of box
        :return:
        """
        self.set_ltwh(left, top, width, height)

    def set_centre_xy(self, xctr: float, yctr: float, width: float, height: float):
        """
        Set bounding box using centre(x,y), width, and height.

        :param xctr: x coordinate of box centre
        :param yctr: y coordinate of box centre
        :param width: width of box
        :param height: height of box
        :return:
        """
        # make sure everything is float
        xctr = float(xctr)
        yctr = float(yctr)
        width = float(width)
        height = float(height)

        self._xctr = xctr
        self._yctr = yctr
        self._width = width
        self._height = height

    def set_centre_yx(self, yctr: float, xctr: float, height: float, width: float):
        """
        Set bounding box using centre(y,x), height, and width.

        :param yctr: y coordinate of box centre
        :param xctr: x coordinate of box centre
        :param height: height of box
        :param width: width of box
        :return:
        """
        self.set_centre_xy(xctr, yctr, width, height)

    @property
    def class_name(self):
        return self._class_name

    @class_name.setter
    def class_name(self, class_name: str):
        self._class_name = class_name

    def get_class_id(self) -> int:
        """
        Get id of class associated with this box

        :return: class id
        """
        return self._class_id

    def get_class_name(self, classes: list[str]) -> str:
        """
        Get the name of the class associated with this box

        :return: class name
        """
        if self.class_name is not None:
            return self.class_name
        return classes[self._class_id]

    def set_class_id(self, class_id: int):
        self._class_id = class_id

    def get_minmax_xy(self) -> tuple[float, float, float, float]:
        xmin = self._xctr - self._width / 2
        ymin = self._yctr - self._height / 2
        xmax = self._xctr + self._width / 2
        ymax = self._yctr + self._height / 2

        return xmin, ymin, xmax, ymax

    def get_minmax_yx(self) -> tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = self.get_minmax_xy()
        return ymin, xmin, ymax, xmax

    def get_ltwh(self) -> tuple[float, float, float, float]:
        left = self._xctr - self._width / 2
        top = self._yctr - self._height / 2
        width = self._width
        height = self._height
        return left, top, width, height

    def get_tlhw(self, img_height: float = None, img_width: float = None) -> tuple[float, float, float, float]:
        """
        Get bounding box in top-left, height, width format.

        :param img_height: (Optional) override stored image height
        :param img_width: (Optional) override stored image width
        :return: a tuple containing top, left, height, width
        """
        left, top, width, height = self.get_ltwh()
        return top, left, height, width

    def get_centre_xy(self) -> tuple[float, float, float, float]:
        """
        Get bounding box in centre(x,y), width, height format.
        :return: a tuple containing xctr, yctr, width, height
        """
        xctr = self._xctr
        yctr = self._yctr
        width = self._width
        height = self._height

        return xctr, yctr, width, height

    def get_centre_yx(self) -> tuple[float, float, float, float]:
        """
        Get bounding box in centre(y,x), height, width format. Can convert between relative and absolute
        representations as indicated by the 'relative' parameter. If conversion is needed, image dimensions must be
        known. This can be done by calling 'set_img_size' before this method, or by setting the 'img_width' or
        'img_height' parameters. The latter will temporarily override any size set through the 'set_img_size' method.

        :return: a tuple containing xctr, yctr, height, width
        """
        xctr, yctr, width, height = self.get_centre_xy()
        return yctr, xctr, height, width

    """
    def _get_overlap(self, bb: 'BoundingBox'):
        # https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        w = max(0, min(self.xmax, bb.xmax) - max(self.xmin, bb.xmin))
        h = max(0, min(self.ymax, bb.ymax) - max(self.ymin, bb.ymin))
        return w * h
    """

    def _calc_intersect_box(self, xmin: float, ymin: float, xmax: float, ymax: float) -> Optional[tuple[float, float, float, float]]:
        this_xmin, this_ymin, this_xmax, this_ymax = self.get_minmax_xy()
        inter_xmin = max(this_xmin, xmin)
        inter_ymin = max(this_ymin, ymin)
        inter_xmax = min(this_xmax, xmax)
        inter_ymax = min(this_ymax, ymax)
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return None
        return inter_xmin, inter_ymin, inter_xmax, inter_ymax

    def resize(self, scale_x: float, scale_y: float) -> 'BoundingBox':
        self.set_minmax_xy(xmin=self.x1 * scale_x, ymin=self.y1 * scale_y, xmax=self.x2 * scale_x, ymax=self.y2 * scale_y)
        return self


def delete_overlapped_bboxes(bboxes: List[BoundingBox], cover_threshold: float) -> List[BoundingBox]:
    if not bboxes:
        return []

    # Sort boxes by area descending (largest first)
    bboxes = sorted(bboxes, key=lambda bb: bb.area, reverse=True)
    keep = []

    for i, current in enumerate(bboxes):
        x1, y1, x2, y2 = current.x1, current.y1, current.x2, current.y2
        keep_box = True

        for other in keep:  # only compare to boxes we've already decided to keep
            xx1 = max(x1, other.x1)
            yy1 = max(y1, other.y1)
            xx2 = min(x2, other.x2)
            yy2 = min(y2, other.y2)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter_area = w * h
            union_area = current.area + other.area - inter_area
            iou = inter_area / (union_area + 1e-6)

            if iou > cover_threshold:
                keep_box = False
                break

        if keep_box:
            keep.append(current)

    return keep


def get_foreground_bboxes(
    img_batch: np.ndarray,
    bboxes: List[BoundingBox],
    area_threshold: float,
    binary_threshold: float,
    gauss_mask_size: int,
) -> List[BoundingBox]:
    """
    Returns new BoundingBoxes for moving foreground regions,
    ignoring given bboxes.
    """
    extend = 2
    img1 = cv2.GaussianBlur(img_batch[0], (gauss_mask_size, gauss_mask_size), 0)
    img2 = cv2.GaussianBlur(img_batch[1], (gauss_mask_size, gauss_mask_size), 0)
    grad = cv2.absdiff(img1, img2)
    sum_grad = cv2.threshold(grad, binary_threshold, 255, cv2.THRESH_BINARY)[1]

    # Mask out given boxes
    for bb in bboxes:
        x1, y1, x2, y2 = map(int, [bb.x1, bb.y1, bb.x2, bb.y2])
        sum_grad[max(0, y1 - extend):min(y2 + extend, sum_grad.shape[0]), max(0, x1 - extend):min(x2 + extend, sum_grad.shape[1])] = 0

    if len(sum_grad.shape) == 3:
        sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: List[BoundingBox] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = (w + 1) * (h + 1)
        if area > area_threshold and w / (h + 1e-6) < 10 and h / (w + 1e-6) < 10:
            bb = BoundingBox()
            bb.set_minmax_xy(
                max(0, x - extend),
                max(0, y - extend),
                min(x + w + extend, sum_grad.shape[1]),
                min(y + h + extend, sum_grad.shape[0]),
            )
            result.append(bb)

    return result


class BoundingBoxProcessor:
    """
    Holds processing functions for bounding box objects.
    """
    def __init__(self, boxes: list[BoundingBox]):
        self.boxes = boxes

    def filter_missing_names(self) -> list[BoundingBox]:
        """
        Filter out boxes that are missing names.

        :return: A list of bounding boxes without boxes with missing names
        """
        new_boxes = []
        for box in self.boxes:
            if box.class_name:
                new_boxes.append(box)
        self.boxes = new_boxes
        return self.boxes

    def filter_area_boxes(self, min_area: int, max_area: int) -> list[BoundingBox]:
        """
        Filter out boxes that are too small and too large.

        :param min_area: Minimum area of box to be considered valid
        :param max_area: Maximum area of box to be considered valid

        :return: A list of bounding boxes without boxes with too small and too large
        """
        new_boxes = []
        for box in self.boxes:
            if min_area < box.area < max_area:
                new_boxes.append(box)
        self.boxes = new_boxes
        return self.boxes

    def resize_boxes(self, scale_x: float, scale_y: float) -> list[BoundingBox]:
        """
        Resize a list of bounding boxes by the given scale factors.

        :param scale_x: Horizontal scaling factor
        :param scale_y: Vertical scaling factor

        :return: List of resized BoundingBox objects
        """
        new_boxes = [bb.resize(scale_x, scale_y) for bb in self.boxes]
        self.boxes = new_boxes
        return self.boxes

    def filter_overlapping_boxes(self, iou_threshold: float = 0.5, box_to_keep: str = Literal["biggest", "smallest"]) -> list[BoundingBox]:
        """
        From overlapping boxes, keep only the smallest one.

        :param iou_threshold: IoU threshold
        :param box_to_keep: Which box to keep in case of overlap, either "biggest" or "smallest"

        :return: A list of filtered BoundingBox objects
        """
        def iou(boxA, boxB):
            # boxA/B expected as (xmin, ymin, xmax, ymax)
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH

            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            union = boxAArea + boxBArea - interArea
            return interArea / union if union > 0 else 0.0

        keep = []
        suppressed = set()

        for i in range(len(self.boxes)):
            if i in suppressed:
                continue

            box_i = self.boxes[i].get_minmax_xy()
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

            for j in range(i + 1, len(self.boxes)):
                if j in suppressed:
                    continue

                box_j = self.boxes[j].get_minmax_xy()
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

                if box_to_keep == "smallest":
                    if iou(box_i, box_j) > iou_threshold:
                        # overlap → keep smaller
                        if area_i < area_j:
                            suppressed.add(j)
                        else:
                            suppressed.add(i)
                elif box_to_keep == "biggest":
                    if iou(box_i, box_j) > iou_threshold:
                        # overlap → keep bigger
                        if area_i > area_j:
                            suppressed.add(j)
                        else:
                            suppressed.add(i)

            if i not in suppressed:
                keep.append(self.boxes[i])

        self.boxes = keep
        return self.boxes

    def match_bboxes(self, boxes2: List[BoundingBox], min_iou: float = 0.5) -> List[int]:
        """
        Match each box in `boxes1` to the best-overlapping box in `boxes2`
        based on IoU (Intersection-over-Union).

        :param boxes2: List of target BoundingBoxes (potential matches)
        :param min_iou: Minimum IoU threshold to consider a valid match
        :return: A list of length len(boxes1), where each element is the index
                 of the matched box in boxes2 (or -1 if no match found)
        """
        if not self.boxes or not boxes2:
            return [-1] * len(self.boxes)

        result = []
        for bb1 in self.boxes:
            x1_1, y1_1, x2_1, y2_1 = bb1.x1, bb1.y1, bb1.x2, bb1.y2
            best_iou = 0.0
            best_idx = -1

            for j, bb2 in enumerate(boxes2):
                x1_2, y1_2, x2_2, y2_2 = bb2.x1, bb2.y1, bb2.x2, bb2.y2

                # compute intersection
                inter_x1 = max(x1_1, x1_2)
                inter_y1 = max(y1_1, y1_2)
                inter_x2 = min(x2_1, x2_2)
                inter_y2 = min(y2_1, y2_2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                area1 = bb1.area
                area2 = bb2.area
                union_area = area1 + area2 - inter_area
                iou = inter_area / (union_area + 1e-6)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            # only keep if above threshold
            if best_iou >= min_iou:
                result.append(best_idx)
            else:
                result.append(-1)

        return result

    def convert_boxes_to_numpy(self) -> np.ndarray:
        if not self.boxes:
            return np.zeros((0, 6), dtype=float)

        boxes_numpy = np.array([np.asarray([box.y1, box.x1, box.y2, box.x2, box.get_class_id(), box.get_confidence()] for box in self.boxes)])

        return boxes_numpy

    def get_boxes(self):
        return self.boxes
