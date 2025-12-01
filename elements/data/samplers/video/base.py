from abc import ABC, abstractmethod


class BaseVideoSampler(ABC):
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    @abstractmethod
    def sample_indices(self, index: int, num_frames: int, frame_video_idx: list[int]) -> list[int]:
        pass
