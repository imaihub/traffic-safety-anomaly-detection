from abc import ABC, abstractmethod

import torch

from elements.data.datatypes.inference_input import InferenceInput
from elements.data.datatypes.inference_result import InferenceResult


class BaseModel(ABC):
    @abstractmethod
    def load_model(self, **args) -> torch.nn.Module:
        """
        Function loading in and returning the instance of the model.
        """
        pass

    @abstractmethod
    def predict(self, x: InferenceInput) -> InferenceResult:
        pass
