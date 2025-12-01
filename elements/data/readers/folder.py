import glob
import os
from typing import Iterator

import cv2
import numpy as np

from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.image.utils.imagecodecs import load_image
from elements.data.readers.base import BaseReader


class FolderReader(BaseReader):
    def __init__(self, dataset_path: str):
        extensions = {".jpg", ".jpeg", ".png"}
        files = [f for f in glob.glob(os.path.join(dataset_path, "*")) if os.path.splitext(f)[1].lower() in extensions]
        self.files = sorted(files)
        self.total_frames = len(self.files)
        self.fps: float = 30

    @property
    def is_stream(self) -> bool:
        return False

    def frames(self, skip_frames: int = 0) -> Iterator[tuple[int, np.ndarray, SampleContainer]]:
        for idx, fpath in enumerate(self.files[skip_frames:], start=skip_frames):
            img = load_image(filename=fpath)
            if img is not None:
                sc = SampleContainer()
                sc.org_image = img
                sc.image_fpath = fpath
                yield idx, img, sc

    def get_frame(self, idx: int) -> tuple[np.ndarray, SampleContainer]:
        img = load_image(self.files[idx])
        if img is None:
            raise ValueError(f"Could not read image {self.files[idx]}")
        sc = SampleContainer()
        sc.org_image = img
        sc.image_fpath = self.files[idx]
        return img, sc

    def release(self):
        pass
