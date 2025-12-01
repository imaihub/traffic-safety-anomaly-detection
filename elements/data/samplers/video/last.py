from elements.data.samplers.video.base import BaseVideoSampler


class LastFrameSampler(BaseVideoSampler):
    """Samples [sequence_length] frames ending at index."""
    def sample_indices(self, index: int, num_frames: int, frame_video_idx: list[int]) -> list[int]:
        start_idx = max(0, index - self.sequence_length)
        indices = list(range(start_idx, index + 1))
        # pad at beginning if not enough frames
        while len(indices) < self.sequence_length + 1:
            indices.insert(0, indices[0])
        return indices
