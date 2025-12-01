from typing import Iterator

import cv2
import numpy as np

from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.readers.base import BaseReader


class VideoReader(BaseReader):
    def __init__(self, dataset_path: str):
        self.reader = cv2.VideoCapture(dataset_path)
        if not self.reader.isOpened():
            raise ValueError(f"Failed to open video file or stream: {dataset_path}")
        self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        self.fps = self.reader.get(cv2.CAP_PROP_FPS) or None

    @property
    def is_stream(self) -> bool:
        return self.total_frames is None

    def frames(self, skip_frames: int = 0) -> Iterator[tuple[int, np.ndarray, SampleContainer]]:
        idx = 0
        success, img = self.reader.read()
        while success:
            if idx >= skip_frames:
                sc = SampleContainer()
                yield idx, img, sc
            idx += 1
            success, img = self.reader.read()

    def get_frame(self, idx: int) -> tuple[np.ndarray, SampleContainer]:
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, img = self.reader.read()
        if not success:
            raise ValueError(f"Could not read frame {idx}")
        sc = SampleContainer()
        return img, sc

    def release(self):
        self.reader.release()
