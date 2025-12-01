import os
from typing import Callable

from elements.data.datatypes.annotations.keypoint import Keypoint
from elements.data.loaders.annotation.utils.keypoints import get_keypoints_cvat

loaders: dict[str, Callable[[str], list[Keypoint]]] = {'.xml': get_keypoints_cvat}


class LoadKeypoints:
    def __call__(self, path: str) -> list[Keypoint]:
        keypoints: list[Keypoint] = []
        for ext, func in loaders.items():
            fn: str = os.path.splitext(path)[0] + ext
            if os.path.isfile(fn):
                with open(fn, "r") as f:
                    keypoints = loaders[ext]("".join(f.readlines()))
            return keypoints
        return keypoints
