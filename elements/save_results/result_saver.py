import os
import shutil
from datetime import datetime
from typing import Optional, Callable

import cv2
import numpy as np

from elements.save_results.video.base import VideoWriterBase
from elements.save_results.video.ffmpeg import FFmpegWriter
from elements.save_results.video.opencv import OpenCVWriter


class VideoResultSaver:
    """
    A VideoResultSaver instance is responsible for saving the results of a video analysis, whether that is loose images or the fully processed video.
    """
    def __init__(
        self,
        output_folder: str = "output",
        save_local_video: bool = False,
        out_file: Optional[str] = None,
        writer_class: Callable = OpenCVWriter,
        writer_kwargs: dict = None,
    ):
        self.local_output_folder = "output"
        self.output_folder = output_folder
        self.save_local_video = save_local_video
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_file = out_file or os.path.join(self.output_folder, "output", "videos", f"{timestamp}.mp4")

        self.writer_class = writer_class
        self.writer_kwargs = writer_kwargs or {}

        self.out_video: VideoWriterBase | None = None
        self.frame_size = None
        self.frames_saved = 0
        self.initialized = False

    def __enter__(self) -> 'VideoResultSaver':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.out_video is not None:
            self.out_video.close()
        if self.save_local_video:
            self.save_copy_video()

    def save_image(self, image: np.ndarray) -> None:
        """
        Save an image to the output folder.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(os.path.join(self.output_folder, 'output', 'images'), exist_ok=True)
        print(f"Saving image to {os.path.join(self.output_folder, 'output', 'images', f'{timestamp}.png')}")
        cv2.imwrite(filename=os.path.join(self.output_folder, "output", "images", f"{timestamp}.png"), img=image)

    def initiate_result_video(self, width: int, height: int, fps: float = 20) -> None:
        """
        Instantiates a custom writer object
        """
        # Proper frame size and codec
        self.frame_size = (width, height)
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
        self.out_video = self.writer_class(path=self.out_file, w=width, h=height, fps=fps, **self.writer_kwargs)
        self.initialized = True

    def append_image_to_video(self, image: np.ndarray) -> None:
        """
        Appends an image to the cv2.VideoWriter instance created using initiate_result_video(...).
        Expects images in RGB format, converts it automatically to BGR so cv2 can save it as RGB
        """
        if not self.initialized:
            self.initiate_result_video(width=image.shape[1], height=image.shape[0], fps=20)

        if isinstance(self.out_video, FFmpegWriter):
            self.out_video.write(frame=cv2.resize(image, (self.frame_size[0], self.frame_size[1])))
        else:
            self.out_video.write(cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), self.frame_size))

        self.frames_saved += 1

    def save_copy_video(self) -> str:
        """
        Copies the fully analyzed video to a local directory.
        """
        if self.frames_saved == 0:
            print("Removing resulting video file as it is empty")
            os.remove(self.out_file)
            return ""

        os.makedirs(os.path.join(self.local_output_folder, "videos"), exist_ok=True)
        local_cache_copy = os.path.join(self.local_output_folder, "videos", os.path.basename(self.out_file))
        shutil.copy(src=self.out_file, dst=local_cache_copy)
        print(f"Saving video to: {local_cache_copy}")
        return local_cache_copy
