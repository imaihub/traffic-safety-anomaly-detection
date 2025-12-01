import cv2

from elements.save_results.video.base import VideoWriterBase


class OpenCVWriter(VideoWriterBase):
    def __init__(self, path, w, h, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.wr = cv2.VideoWriter(path, fourcc, fps, (w, h))

    def write(self, frame):
        self.wr.write(frame)

    def close(self):
        self.wr.release()
