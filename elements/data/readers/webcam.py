import os
from datetime import datetime
from typing import Optional, Iterator

import cv2
import numpy as np
from cv2 import VideoCapture

from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.readers.base import BaseReader
from elements.data.transforms.preprocess.image import Compose

# Common resolutions to try
possible_resolutions = {
    (640, 480): False,
    (1024, 768): False,
    (1280, 720): False,
    (1920, 1080): False,
}


def get_webcam_settings(camera_index: int = -1, verbose: bool = False) -> tuple[VideoCapture, tuple[int, int], int] | None:
    """
    Tests webcam settings automatically to get valid settings, namely the camera index and maximum valid resolution.

    :param camera_index: If manually set, this is the camera index to use. -1 means automatic
    :param verbose: Give a summary of possible webcam settings
    :return: Camera index to use, and the maximum valid resolution.

    """
    def create_video_capture(camera_index: int, resolution: tuple) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index, apiPreference=cv2.CAP_ANY)

        # Force MJPG (compressed, supports higher res)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        return cap

    cap = None
    maximum_resolution = None

    # Check whether user input camera index is valid
    if camera_index >= 0:
        cap = cv2.VideoCapture(camera_index)

        ok, frame = cap.read()
        if not ok:
            print(f"Camera index {camera_index} is not a valid camera index.")
            return None

    if camera_index < 0:
        # Determine usable camera index first
        for i in range(4):
            cap = cv2.VideoCapture(i)

            ok, frame = cap.read()
            if not ok:
                continue

            camera_index = i

    if camera_index < 0:
        raise ValueError("No valid camera index found, camera mode not possible")

    # Loop over possible resolutions to check which is valid
    for resolution in possible_resolutions.keys():
        if isinstance(cap, cv2.VideoCapture):
            cap.release()

        cap = create_video_capture(camera_index, resolution)

        ok, frame = cap.read()
        if ok and frame.shape[:2] == resolution[::-1]:
            possible_resolutions[resolution] = True
            maximum_resolution = resolution
            continue

    if maximum_resolution is None:
        print("No valid resolution found, camera mode not possible")
        return None

    if verbose:
        print("Possible webcam settings:")
        for resolution in possible_resolutions:
            print(f"Resolution: {resolution}: {'Yes' if possible_resolutions[resolution] else 'No'}")

    if cap is not None:
        cap.release()

    cap = create_video_capture(camera_index, maximum_resolution)

    return cap, maximum_resolution, camera_index


class WebcamReader(BaseReader):
    def __init__(self, camera_index: int = -1, save_all_frames: bool = False, save_directory: str = "output/raw_frames", transforms: Optional[Compose] = None, verbose: bool = False):
        self.reader, self.camera_index, self.resolution = get_webcam_settings(camera_index, verbose)
        self.total_frames = None
        self.fps = 30
        self.is_streaming = True
        self.transforms = transforms
        self.save_all_frames = save_all_frames

        if save_all_frames:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_folder = os.path.join(save_directory, timestamp)
            os.makedirs(self.save_folder, exist_ok=True)

    @property
    def is_stream(self) -> bool:
        return True

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
        raise NotImplementedError("WebcamReader does not support random access")

    def release(self):
        self.reader.release()
