from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm


class FaissModel:
    def __init__(self, feature_key: str = "features"):
        self.res = faiss.StandardGpuResources()
        self.first_feature_added = False
        self.index = None
        self.index_gpu = None
        self.min = np.inf
        self.max = -np.inf
        self.dim = None
        self.feature_key = feature_key

    def set_calibration_values(self, min: float, max: float):
        self.min = min
        self.max = max

    @staticmethod
    def load_numpy_file(file_path: str):
        arr = np.load(file_path, allow_pickle=True)
        if hasattr(arr, "item"):
            arr = arr.item()
        return arr

    def load_feature_from_file(self, file_path: str):
        arr = np.load(file_path, allow_pickle=True)
        try:
            arr = arr.item()[self.feature_key].astype(np.float32)
        except:
            arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        return arr

    def add_first_feature(self, feature: np.ndarray):
        self.dim = feature.shape[-1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index_gpu = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        self.first_feature_added = True

    def add_feature(self, feature: np.ndarray):
        self.index_gpu.add(feature)

    def add_features_from_files(self, files: list):
        for f in files:
            feature = self.load_feature_from_file(f)
            if not feature.any():
                continue
            if not self.first_feature_added:
                self.add_first_feature(feature=feature)
                continue
            self.add_feature(feature=feature)

    def add_features_from_arrays(self, features: np.ndarray):
        for feature in features:
            if not self.first_feature_added:
                self.add_first_feature(feature=feature)
                continue
            self.add_feature(feature=feature)

    def calibrate(self, files: list, knn: int = 2):
        for f in tqdm(files, desc="Calibrating deep scores"):
            arr = self.load_feature_from_file(f)
            if not arr.any():
                continue
            D, _ = self.index_gpu.search(arr, knn)
            scores = np.mean(D, axis=1)
            self.min = min(self.min, scores.min())
            self.max = max(self.max, np.percentile(scores, 99.9))

    def get_score(self, knn: int, raw: bool = False, data: Optional[np.ndarray] = None, file_path: str = None, take_max: bool = False, normalize: bool = True) -> float:
        if data is not None:
            pass
        elif file_path is not None:
            data = self.load_feature_from_file(file_path)
        else:
            raise NotImplementedError("No feature array or valid file path given")
        if not data.any():
            return np.array(0, dtype=np.float32)
        D, _ = self.index_gpu.search(data, knn)
        scores = np.mean(D, axis=1)
        if raw:
            return D
        if take_max:
            score = np.max(scores)
        else:
            score = scores
        if normalize:
            score = (score - self.min) / (self.max - self.min)
        return score
