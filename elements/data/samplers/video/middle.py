from elements.data.samplers.video.base import BaseVideoSampler


class MiddleFrameSampler(BaseVideoSampler):
    """Samples symmetric window around index."""
    def sample_indices(self, index: int, num_frames: int, frame_video_idx: list[int]) -> list[int]:
        half = self.sequence_length
        start = max(0, index - half)
        end = min(num_frames - 1, index + half)
        indices = list(range(start, end + 1))
        # pad if too short
        while len(indices) < (2 * half + 1):
            if start == 0:
                indices.insert(0, indices[0])
            else:
                indices.append(indices[-1])
        return indices
