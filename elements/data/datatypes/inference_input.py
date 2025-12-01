from dataclasses import dataclass

import numpy as np
import torch

from elements.data.datatypes.samplecontainer import SampleContainer


@dataclass
class InferenceInput:
    sc: list[SampleContainer] | SampleContainer
    threshold: float = 0.15
