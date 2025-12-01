from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


class GMMModel:
    def __init__(self, gmm: int, random_state: int = 0):
        self.n_components = gmm
        self.gmm_model = GaussianMixture(n_components=gmm, random_state=random_state, warm_start=True)
        self.scores = None
        self.min = np.inf
        self.max = -np.inf

    def set_calibration_values(self, min: float, max: float):
        self.min = min
        self.max = max

    @staticmethod
    def load_numpy_file(file_path: str):
        arr = np.load(file_path, allow_pickle=True)
        if hasattr(arr, "item"):
            arr = arr.item()
        return arr

    @staticmethod
    def load_feature_from_file(file_path: str, expand: bool = True):
        arr = np.load(file_path, allow_pickle=True)
        if hasattr(arr, "item"):
            arr = arr.item()["velocities"].astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        if expand:
            if arr.ndim == 1:
                arr = arr[None]
        return arr

    def calibrate(self, train_files: list):
        print("Calibrate GMM scores (min/max)")
        for f in tqdm(train_files, desc="Calibrating velocity scores"):
            arr = self.load_feature_from_file(file_path=f, expand=False)
            if not arr.any():
                continue
            self.scores = -self.gmm_model.score_samples(arr)
            self.min = min(self.min, self.scores.min())
            self.max = max(self.max, np.percentile(self.scores, 99.9))

    def train(self, train_files: list, batch_size: int, calibrate: bool = True):
        print("Training GMM...")
        batch_feats = []
        for i, f in enumerate(train_files):
            arr = self.load_feature_from_file(f)
            if arr.any():
                batch_feats.append(arr)
            if len(batch_feats) == batch_size:
                batch_feats = np.concatenate(batch_feats, 0)
                self.gmm_model.fit(batch_feats)
                batch_feats = []
        if len(batch_feats) > self.n_components:  # Left over data
            batch_feats = np.concatenate(batch_feats, 0)
            self.gmm_model.fit(batch_feats)

        if calibrate:
            self.calibrate(train_files=train_files)

    def get_score(self, data: Optional[np.ndarray] = None, file_path: Optional[str] = None, take_max: bool = False, normalize: bool = True) -> float:
        if data is not None:
            pass
        elif file_path is not None:
            data = self.load_feature_from_file(file_path)
        else:
            raise NotImplementedError("No feature array or valid file path given")
        if not data.any():
            return np.array(0, dtype=np.float32)
        if take_max:
            score = np.max(-self.gmm_model.score_samples(data))
        else:
            score = -self.gmm_model.score_samples(data)
        if normalize:
            score = (score - self.min) / (self.max - self.min)
        return score
