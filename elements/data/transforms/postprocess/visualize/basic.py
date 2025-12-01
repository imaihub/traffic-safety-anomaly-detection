import cv2
import numpy as np
from typing import Optional

from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.keypoint import Keypoint

# ----------------------------
# Individual visualization funcs
# ----------------------------


def draw_text_with_box(frame, text, position, color, is_dashed=False, padding=5):
    """Pre-calculate text measurements"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    x, y = position
    box_start = (x - padding, y - text_height - padding)
    box_end = (x + text_width + padding, y + padding)

    # Use numpy operations for rectangle filling
    y1, y2 = box_start[1], box_end[1]
    x1, x2 = box_start[0], box_end[0]
    frame[y1:y2, x1:x2] = [255, 255, 255]

    if is_dashed:
        draw_dashed_rectangle_gap(frame, box_start, box_end, color, 2, gap=5)
    else:
        cv2.rectangle(frame, box_start, box_end, color, 2)

    cv2.putText(frame, text, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)


def draw_dashed_rectangle_gap(frame, start_point, end_point, color, thickness, gap=10):
    """Optimized drawing using vectorized operations"""
    x1, y1 = start_point
    x2, y2 = end_point

    # Pre-calculate points for all lines
    h_points = np.array([(i, i + gap) for i in range(x1, x2, gap * 2)])
    v_points = np.array([(i, i + gap) for i in range(y1, y2, gap * 2)])

    # Draw horizontal lines
    for y in (y1, y2):
        for start, end in h_points:
            end = min(end, x2)
            cv2.line(frame, (start, y), (end, y), color, thickness)

    # Draw vertical lines
    for x in (x1, x2):
        for start, end in v_points:
            end = min(end, y2)
            cv2.line(frame, (x, start), (x, end), color, thickness)


def draw_dashed_rectangle(img: np.ndarray, start_point: tuple[int, int], end_point: tuple[int, int], color: tuple[int, int, int], thickness: float = 0.5, dash_length: int = 5):
    """Vectorized dashed rectangle drawing"""
    x1, y1 = start_point
    x2, y2 = end_point

    # Pre-calculate dash positions
    h_dashes = np.arange(x1, x2, dash_length * 2)
    v_dashes = np.arange(y1, y2, dash_length * 2)

    # Vectorized horizontal lines
    for y in (y1, y2):
        ends = np.minimum(h_dashes + dash_length, x2)
        for start, end in zip(h_dashes, ends):
            cv2.line(img, (int(start), y), (int(end), y), color, int(thickness))

    # Vectorized vertical lines
    for x in (x1, x2):
        ends = np.minimum(v_dashes + dash_length, y2)
        for start, end in zip(v_dashes, ends):
            cv2.line(img, (x, int(start)), (x, int(end)), color, int(thickness))


def draw_boxes_custom(img: np.ndarray, boxes: list[BoundingBox], text: str, color: tuple[int, int, int], ratios: Optional[tuple[float, float]] = None, thickness: int = 2, dashed_boxes: bool = False) -> np.ndarray:
    vis = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(round, [box.x1, box.y1, box.x2, box.y2])
        if ratios:
            x1, x2 = int(x1 * ratios[0]), int(x2 * ratios[0])
            y1, y2 = int(y1 * ratios[1]), int(y2 * ratios[1])
        if dashed_boxes:
            draw_dashed_rectangle(vis, (x1, y1), (x2, y2), color, thickness, gap=5)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(vis, text, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def draw_boxes(img: np.ndarray, boxes: list[BoundingBox], color_map: Optional[dict[int, tuple[int, int, int]]] = None, ratios: Optional[tuple[float, float]] = None, thickness: int = 2, dashed_boxes: bool = False, text: bool = True) -> np.ndarray:
    vis = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(round, [box.x1, box.y1, box.x2, box.y2])
        if ratios:
            x1, x2 = int(x1 * ratios[0]), int(x2 * ratios[0])
            y1, y2 = int(y1 * ratios[1]), int(y2 * ratios[1])
        color = (0, 255, 0) if color_map is None else color_map.get(box.get_class_id(), (0, 255, 0))
        if dashed_boxes:
            draw_dashed_rectangle(vis, (x1, y1), (x2, y2), color, thickness, gap=5)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if text:
            cv2.putText(vis, f"{box.get_confidence():.2f}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def draw_keypoints(img: np.ndarray, keypoints: list[Keypoint], color: tuple[int, int, int] = (0, 255, 0), radius: int = 3, thickness: int = -1) -> np.ndarray:
    vis = img.copy()
    for kp in keypoints:
        x, y = int(kp.x), int(kp.y)
        cv2.circle(vis, (x, y), radius, color, thickness)
    return vis


def draw_fps(img: np.ndarray, fps: float, position: tuple[int, int] = (50, 50), color: tuple[int, int, int] = (40, 255, 255)) -> np.ndarray:
    vis = img.copy()
    cv2.putText(vis, f"FPS: {fps:.1f}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


def draw_progress(img: np.ndarray, progress: float, progress_text: Optional[str] = None, bar_color: tuple[int, int, int] = (0, 255, 0), bg_color: tuple[int, int, int] = (50, 50, 50), text_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    progress = np.clip(progress, 0, 100)
    text = progress_text or f"{progress:.0f}%"

    bar_width = int(0.7 * w)
    bar_height = int(0.05 * h)
    bar_x = 50
    bar_y = int(h - bar_height - int(0.05 * h))

    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), bg_color, -1)
    progress_width = int((progress / 100.0) * bar_width)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int(bar_x + (bar_width / 2) - (text_size[0] / 2))
    text_y = bar_y - 10
    cv2.putText(vis, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return vis


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    # Convert CHW -> HWC if needed
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))

    h, w, _ = flow.shape
    hsv_mask = np.zeros((h, w, 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 1] = 255
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)
    return rgb


# ----------------------------
# Compose function
# ----------------------------
def visualize(
    img: np.ndarray,
    boxes: Optional[list[BoundingBox]] = None,
    keypoints: Optional[list] = None,
    fps: Optional[float] = None,
    progress: Optional[float] = None,
    progress_text: Optional[str] = None,
    flow: Optional[np.ndarray] = None,
    color_map: Optional[dict[int, tuple[int, int, int]]] = None,
    ratios: Optional[tuple[float, float]] = None,
    dashed_boxes: bool = False,
) -> np.ndarray:
    vis = img.copy()
    if boxes: vis = draw_boxes(vis, boxes, color_map=color_map, ratios=ratios, dashed_boxes=dashed_boxes)
    if keypoints: vis = draw_keypoints(vis, keypoints)
    if fps is not None: vis = draw_fps(vis, fps)
    if progress is not None: vis = draw_progress(vis, progress, progress_text=progress_text)
    if flow is not None: vis = flow_to_rgb(flow)
    return vis
