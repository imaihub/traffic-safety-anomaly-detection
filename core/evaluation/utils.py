import sys
from copy import deepcopy

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score


def gaussian_video(scores: dict[str, list[np.ndarray]], sigma=3) -> dict[str, np.ndarray]:
    """Applies a 1D Gaussian smoothing filter to anomaly scores.

    Args:
        scores (dict[str, np.ndarray]): Final anomaly scores to be smoothed.
        sigma (int, optional): Standard deviation for the Gaussian kernel.

    Returns:
        scores (dict[str, np.ndarray]): Original scores with additional key-value pair with key "{video}_smoothed".
    """
    scores_copy = deepcopy(scores)
    keys = list(scores_copy.keys())
    for video in keys:
        if "smooth" not in video and "boxes" not in video:
            if isinstance(scores_copy[video], list):
                # HAVE TO FIX THIS
                for idx in range(len(scores_copy[video])):
                    if not scores_copy[video][idx].any():
                        scores_copy[video][idx] = np.array(0, dtype=np.float32)

                scores_copy[video] = np.array(scores_copy[video])
            scores_copy[f"{video}_smoothed"] = gaussian_filter1d(scores_copy[video], sigma)
    return scores_copy


def gaussian_video(scores: dict[str, list[np.ndarray]], sigma=3, replacement=None) -> dict[str, np.ndarray]:
    """Applies a 1D Gaussian smoothing filter to anomaly scores.

    Args:
        scores (dict[str, np.ndarray]): Final anomaly scores to be smoothed.
        sigma (int, optional): Standard deviation for the Gaussian kernel.

    Returns:
        scores (dict[str, np.ndarray]): Original scores with additional key-value pair with key "{video}_smoothed".
    """
    smoothed_scores = deepcopy(scores)
    keys = list(smoothed_scores.keys())
    for video in keys:
        if isinstance(smoothed_scores[video], list):
            for idx in range(len(smoothed_scores[video])):
                if not smoothed_scores[video][idx].any():
                    if replacement is not None:
                        smoothed_scores[video][idx] = replacement
                    else:
                        smoothed_scores[video][idx] = np.array(0, dtype=np.float32)
            smoothed_scores[video] = gaussian_filter1d(smoothed_scores[video], sigma)
    return smoothed_scores


def macro_auc(scores: np.ndarray, lengths: list[int], test_labels: list[int]):
    """Computes the Macro AUC-ROC.

    The function calculates the AUC-ROC per clip and then
    averages these values across all clips.

    Args:
        scores (np.ndarray): Final anomaly scores.
        lengths (list[int]): Clip lengths.
        test_labels (np.ndarray): Ground-truth labels (0 for normal, 1 for anomaly).

    Returns:
        Macro AUC-ROC (float): The averaged Macro AUC-ROC across all segments.
    """
    auc = roc_auc_score(np.concatenate(([0], test_labels, [1])), np.concatenate(([0], scores, [sys.float_info.max])))
    return auc
