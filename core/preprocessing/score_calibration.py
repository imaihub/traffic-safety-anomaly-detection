import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config.paths import LoadPaths
from elements.config.load_config import Config
from elements.data.utils import prune_lists_smallest_length, get_file_paths
from elements.enums.enums import FileExtension
from elements.load_model.extractors.faiss_model import FaissModel
from elements.load_model.extractors.gmm import GMMModel

def compute_calibration_parameters(dataset_name: str, max_image_count: int = -1, batch_size: int = 50, gmm: int = 5):
    dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="training")

    deep_features_paths = get_file_paths(dirs.DEEP_FEATURES, recursive=True, max_count=max_image_count, extensions=[FileExtension.NPY])
    keypoints_paths = get_file_paths(dirs.KEYPOINTS, recursive=True, max_count=max_image_count, extensions=[FileExtension.NPY])
    vel_paths = get_file_paths(dirs.VELOCITIES, recursive=True, max_count=max_image_count, extensions=[FileExtension.NPY])

    deep_features_paths, keypoints_paths = prune_lists_smallest_length([deep_features_paths, keypoints_paths])

    print(f"Starting score calibration on dataset {dataset_name}")
    faiss_model_deep_features = None
    faiss_model_keypoints = None

    gmm_model = GMMModel(gmm=gmm)
    gmm_model.train(train_files=vel_paths, batch_size=batch_size)
    gmm_model.calibrate(train_files=vel_paths)

    current_video = None

    min_deep, max_deep = np.inf, -np.inf
    min_keypoints, max_keypoints = np.inf, -np.inf

    for deep_features_path, keypoints_path in zip(deep_features_paths, keypoints_paths):
        cv = deep_features_path.split(os.sep)[-2]
        if current_video is None or not cv == current_video:
            current_video = cv
            rest_deep_feature_files = [path for path in deep_features_paths if not path.split(os.sep)[-2] == cv]
            rest_keypoints_files = [path for path in keypoints_paths if not path.split(os.sep)[-2] == cv]

            faiss_model_deep_features = FaissModel("features")
            faiss_model_deep_features.add_features_from_files(files=rest_deep_feature_files)

            faiss_model_keypoints = FaissModel("keypoints")
            faiss_model_keypoints.add_features_from_files(files=rest_keypoints_files)

        print(f"Processing {deep_features_path}")

        D_deep_features = faiss_model_deep_features.get_score(file_path=deep_features_path, knn=2, raw=True)
        D_keypoints = faiss_model_keypoints.get_score(file_path=keypoints_path, knn=2, raw=True)
        if len(D_deep_features.shape) <= 1 or len(D_keypoints.shape) <= 1:
            frame_scores_deep_features = np.array([0.0], dtype=np.float32)
            frame_scores_keypoints = np.array([0.0], dtype=np.float32)
        else:
            # k=2 → first is self, second is closest neighbor
            frame_scores_deep_features = D_deep_features[:, 1]
            frame_scores_keypoints = D_keypoints[:, 1]

        if np.any(frame_scores_keypoints):
            max_keypoints = max(max_keypoints, np.percentile(frame_scores_keypoints, 99))
            min_keypoints = min(min_keypoints, np.min(frame_scores_keypoints))

        if np.any(frame_scores_deep_features):
            max_deep = max(max_deep, np.percentile(frame_scores_deep_features, 99))
            min_deep = min(min_deep, np.min(frame_scores_deep_features))

    with Config() as config_writer:
        config_writer.add_field("calibration", {
            "min_deep": float(min_deep),
            "max_deep": float(max_deep),
            "min_keypoints": float(min_keypoints),
            "max_keypoints": float(max_keypoints),
            "min_vel": float(gmm_model.min),
            "max_vel": float(gmm_model.max),
        })

    print(f"Calibration parameters saved to {config_writer.path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech", help='Name given to dataset')
    parser.add_argument("--gmm", type=int, default=5)
    parser.add_argument("--max-image-count", type=int, default=4000, help='Limit amount of frames for processing')
    args = parser.parse_args()

    compute_calibration_parameters(dataset_name=args.dataset_name, max_image_count=args.max_image_count, batch_size=50, gmm=args.gmm)
