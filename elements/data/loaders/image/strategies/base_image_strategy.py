from abc import ABC, abstractmethod

import numpy as np


class BaseImageStrategy(ABC):
    @staticmethod
    @abstractmethod
    def get_numpy_array_from_path(path: str) -> np.ndarray:
        pass