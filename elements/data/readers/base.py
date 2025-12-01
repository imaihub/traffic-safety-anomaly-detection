from abc import abstractmethod, ABC
from typing import Iterator, Optional

import numpy as np

from elements.data.datatypes.samplecontainer import SampleContainer


class BaseReader(ABC):
    total_frames: Optional[int]  # None if unknown/infinite
    fps: Optional[float]
    logger_interval = 100  # Every 100th iteration, give update

    @abstractmethod
    def frames(self, skip_frames: int = 0) -> Iterator[tuple[int, np.ndarray, SampleContainer]]:
        pass

    @abstractmethod
    def get_frame(self, idx: int) -> tuple[np.ndarray, SampleContainer]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @property
    def is_stream(self) -> bool:
        return self.total_frames is None
