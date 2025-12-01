import glob
import os
from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np

from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.image.utils.imagecodecs import load_image
from elements.data.readers.base import BaseReader
from elements.data.utils import get_file_paths
from elements.enums.enums import FileExtension


class MultiVideoImageReader(BaseReader):
    def __init__(
        self,
        dataset_path: str,
        max_image_count: Optional[int] = None  # new parameter
    ):
        self.videos = OrderedDict()
        self.frame_addresses = []
        self.frame_video_idx = []

        for i, video_path in enumerate(sorted(glob.glob(os.path.join(dataset_path, "*")))):
            if max_image_count is not None and max_image_count > 0:
                if len(self.frame_addresses) > max_image_count:
                    break

            frames = get_file_paths(directory=video_path, extensions=[FileExtension.JPG, FileExtension.PNG, FileExtension.JPEG], recursive=False, sort_list=True)
            self.videos[i] = {
                "path": video_path,
                "frames": frames,
                "length": len(frames),
            }
            self.frame_addresses.extend(frames)
            self.frame_video_idx.extend([i] * len(frames))

        if max_image_count is not None and max_image_count > 0:
            self.frame_addresses = self.frame_addresses[:max_image_count]
            self.frame_video_idx = self.frame_video_idx[:max_image_count]
        self.total_frames = len(self.frame_addresses)
        self.fps = 30

    @property
    def is_stream(self) -> bool:
        return False

    def frames(self, skip_frames: int = 0) -> Iterator[tuple[int, np.ndarray, SampleContainer]]:
        for idx, fpath in enumerate(self.frame_addresses[skip_frames:], start=skip_frames):
            if idx % self.logger_interval == 0:
                print(f"Processing frame {idx} of {self.total_frames} with path {fpath}...")
            img = load_image(filename=fpath)
            if img is not None:
                sc = SampleContainer()
                sc.image_fpath = fpath
                yield idx, img, sc

    def get_frame(self, idx: int) -> tuple[np.ndarray, SampleContainer]:
        img = load_image(self.frame_addresses[idx])
        if img is None:
            raise ValueError(f"Could not read frame {self.frame_addresses[idx]}")
        sc = SampleContainer()
        sc.image_fpath = self.frame_addresses[idx]
        return img, sc

    def release(self):
        pass
