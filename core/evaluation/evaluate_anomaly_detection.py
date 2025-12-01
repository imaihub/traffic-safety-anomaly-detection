import argparse
import os
import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from core.config.paths import LoadPaths
from elements.config.load_config import Config
from core.evaluation.utils import gaussian_video
from core.utils.evaluator import ObjectLevelEvaluator
from elements.data.utils import prune_lists_smallest_length, concatenate_flatten_arrays, get_file_paths, get_file_path
from elements.enums.enums import FileExtension
from elements.load_model.extractors.faiss_model import FaissModel
from elements.load_model.extractors.gmm import GMMModel

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def create_roc_curve(path: str, ground_truths: np.ndarray, final_scores: np.ndarray, micro_auc: float):
    plt.figure()
    fpr, tpr, _ = roc_curve(ground_truths, final_scores)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {micro_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Frame/Object-level ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def evaluate(batch_size: int, gmm: int, knn: int, sigma: int, dataset_name: str, max_image_count_training: int, max_image_count_testing: int):
    """Evaluation of the video anomaly detection"""
    train_dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="training")
    test_dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="testing")

    train_vel_files = get_file_paths(train_dirs.VELOCITIES, [FileExtension.NPY], recursive=True, max_count=max_image_count_training)
    train_deep_files = get_file_paths(train_dirs.DEEP_FEATURES, [FileExtension.NPY], recursive=True, max_count=max_image_count_training)
    train_keypoints_files = get_file_paths(train_dirs.KEYPOINTS, [FileExtension.NPY], recursive=True, max_count=max_image_count_training)

    test_vel_files = get_file_paths(test_dirs.VELOCITIES, [FileExtension.NPY], recursive=True, max_count=max_image_count_testing)
    test_deep_files = get_file_paths(test_dirs.DEEP_FEATURES, [FileExtension.NPY], recursive=True, max_count=max_image_count_testing)
    test_keypoints_files = get_file_paths(test_dirs.KEYPOINTS, [FileExtension.NPY], recursive=True, max_count=max_image_count_testing)

    test_vel_files, test_deep_files, test_keypoints_files = prune_lists_smallest_length([test_vel_files, test_deep_files, test_keypoints_files])

    config = Config()
    min_keypoints = config.get("calibration").get("min_keypoints")
    max_keypoints = config.get("calibration").get("max_keypoints")
    min_deep = config.get("calibration").get("min_deep")
    max_deep = config.get("calibration").get("max_deep")
    min_velocity = config.get("calibration").get("min_vel")
    max_velocity = config.get("calibration").get("max_vel")

    # Train and calibrate GMM Model
    gmm_model = GMMModel(gmm=gmm)
    gmm_model.train(train_files=train_vel_files, batch_size=batch_size)
    gmm_model.set_calibration_values(min=min_velocity, max=max_velocity)

    # Train and calibrate Faiss Model for deep features
    faiss_model_features = FaissModel(feature_key="features")
    faiss_model_features.add_features_from_files(files=train_deep_files)
    faiss_model_features.set_calibration_values(min=min_deep, max=max_deep)

    # Train and calibrate Faiss Model for keypoints
    faiss_model_keypoints = FaissModel(feature_key="keypoints")
    faiss_model_keypoints.add_features_from_files(files=train_keypoints_files)
    faiss_model_keypoints.set_calibration_values(min=min_keypoints, max=max_keypoints)

    print("Collecting scores for testing set...")
    velocity_scores_frames, deep_scores_frames, keypoints_scores_frames, combined_scores_frames = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    velocity_scores_objects, deep_scores_objects, keypoints_scores_objects, combined_scores_objects = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    predicted_boxes = defaultdict(list)
    current_video = None
    for idx, (kf, vf, df) in tqdm(enumerate(zip(test_keypoints_files, test_vel_files, test_deep_files)), total=len(test_vel_files), desc="Testing"):
        cv = os.path.dirname(vf)
        if current_video is None or not cv == current_video:
            current_video = cv

        npy_file = GMMModel.load_numpy_file(vf)
        boxes = npy_file["boxes"]

        # Velocity score frame level
        v_score_frame = gmm_model.get_score(file_path=vf, take_max=True, normalize=True)
        velocity_scores_frames[current_video].append(v_score_frame)

        # Deep score frame level
        d_score_frame = faiss_model_features.get_score(file_path=df, knn=knn, take_max=True, normalize=True)
        deep_scores_frames[current_video].append(d_score_frame)

        # Keypoints score frame level
        k_score_frame = faiss_model_keypoints.get_score(file_path=kf, knn=knn, take_max=True, normalize=True)
        keypoints_scores_frames[current_video].append(k_score_frame)

        # Velocity score object level
        v_score_objects = gmm_model.get_score(file_path=vf, normalize=True)
        velocity_scores_objects[current_video].append(v_score_objects)

        # Deep score object level
        d_score_objects = faiss_model_features.get_score(file_path=df, knn=knn, normalize=True)
        deep_scores_objects[current_video].append(d_score_objects)

        # Keypoints score object level
        k_score_objects = faiss_model_keypoints.get_score(file_path=kf, knn=knn, normalize=True)
        keypoints_scores_objects[current_video].append(k_score_objects)

        combined_score_frame = (v_score_frame + d_score_frame + k_score_frame) / 3
        combined_scores_frames[current_video].append(combined_score_frame)

        combined_score_objects = (v_score_objects + d_score_objects + k_score_objects) / 3
        combined_scores_objects[current_video].append(combined_score_objects)

        predicted_boxes[current_video].append(boxes)

    # Frame level scores
    smoothed_scores_frames_velocity = gaussian_video(scores=velocity_scores_frames, sigma=sigma)
    smoothed_scores_frames_deep = gaussian_video(scores=deep_scores_frames, sigma=sigma)
    smoothed_scores_frames_keypoints = gaussian_video(scores=keypoints_scores_frames, sigma=sigma)
    smoothed_scores_frames_combined = gaussian_video(scores=combined_scores_frames, sigma=sigma)

    # Handle ground truth frame level
    gt_frame_files = get_file_paths(test_dirs.FRAME_MASK_GT, [FileExtension.NPY], recursive=True)

    # Flatten velocity and deep scores from all subdirectories
    final_scores_frames_velocity_all = []
    for directory in smoothed_scores_frames_velocity.keys():
        final_scores_frames_velocity_all.append(smoothed_scores_frames_velocity[directory])
    final_scores_frames_velocity_all = np.concatenate(final_scores_frames_velocity_all, axis=0)

    final_scores_frames_deep_all = []
    for directory in smoothed_scores_frames_deep.keys():
        final_scores_frames_deep_all.append(smoothed_scores_frames_deep[directory])
    final_scores_frames_deep_all = np.concatenate(final_scores_frames_deep_all, axis=0)

    final_scores_frames_keypoints_all = []
    for directory in smoothed_scores_frames_keypoints.keys():
        final_scores_frames_keypoints_all.append(smoothed_scores_frames_keypoints[directory])
    final_scores_frames_keypoints_all = np.concatenate(final_scores_frames_keypoints_all, axis=0)

    final_scores_frames_combined_all = []
    for directory in smoothed_scores_frames_combined.keys():
        final_scores_frames_combined_all.append(smoothed_scores_frames_combined[directory])
    final_scores_frames_combined_all = np.concatenate(final_scores_frames_combined_all, axis=0)

    ground_truths = concatenate_flatten_arrays(file_paths=gt_frame_files)[:len(final_scores_frames_velocity_all)]

    micro_auc_velocity = roc_auc_score(ground_truths, final_scores_frames_velocity_all)
    micro_auc_deep = roc_auc_score(ground_truths, final_scores_frames_deep_all)
    micro_auc_keypoints = roc_auc_score(ground_truths, final_scores_frames_keypoints_all)
    micro_auc_combined = roc_auc_score(ground_truths, final_scores_frames_combined_all)

    print(f"Micro AUC Velocity: {micro_auc_velocity * 100:.2f}")
    print(f"Micro AUC Deep: {micro_auc_deep * 100:.2f}")
    print(f"Micro AUC Keypoints: {micro_auc_keypoints * 100:.2f}")
    print(f"Micro AUC Combined: {micro_auc_combined * 100:.2f}")

    # Handle object level predictions and match them with the ground truths
    gt_mask_paths = get_file_paths(directory=test_dirs.PIXEL_MASK_GT, extensions=[FileExtension.NPY], recursive=True)
    object_evaluator = ObjectLevelEvaluator(gt_mask_paths=gt_mask_paths, velocity_scores_objects=velocity_scores_objects, deep_scores_objects=deep_scores_objects, keypoints_scores_objects=keypoints_scores_objects, combined_scores_objects=combined_scores_objects, predicted_boxes=predicted_boxes)
    results = object_evaluator.run()
    object_predictions_per_frame = results["object_predictions_per_frame"]

    # Use ROC-based threshold for final predictions on frame-level
    fpr_velocity_frames, tpr_velocity_frames, thresholds_velocity_frames = roc_curve(ground_truths, final_scores_frames_velocity_all)
    youden_idx_velocity_frames = np.argmax(tpr_velocity_frames - fpr_velocity_frames)
    optimal_threshold_velocity_frames = thresholds_velocity_frames[youden_idx_velocity_frames]
    predictions_velocity_frames = (final_scores_frames_velocity_all >= optimal_threshold_velocity_frames).astype(int)

    fpr_deep_frames, tpr_deep_frames, thresholds_deep_frames = roc_curve(ground_truths, final_scores_frames_deep_all)
    youden_idx_deep_frames = np.argmax(tpr_deep_frames - fpr_deep_frames)
    optimal_threshold_deep_frames = thresholds_deep_frames[youden_idx_deep_frames]
    predictions_deep_frames = (final_scores_frames_deep_all >= optimal_threshold_deep_frames).astype(int)

    fpr_keypoints_frames, tpr_keypoints_frames, thresholds_keypoints_frames = roc_curve(ground_truths, final_scores_frames_keypoints_all)
    youden_idx_keypoints_frames = np.argmax(tpr_keypoints_frames - fpr_keypoints_frames)
    optimal_threshold_keypoints_frames = thresholds_keypoints_frames[youden_idx_keypoints_frames]
    predictions_keypoints_frames = (final_scores_frames_keypoints_all >= optimal_threshold_keypoints_frames).astype(int)

    fpr_combined_frames, tpr_combined_frames, thresholds_combined_frames = roc_curve(ground_truths, final_scores_frames_combined_all)
    youden_idx_combined_frames = np.argmax(tpr_combined_frames - fpr_combined_frames)
    optimal_threshold_combined_frames = thresholds_combined_frames[youden_idx_combined_frames]
    predictions_combined_frames = (final_scores_frames_combined_all >= optimal_threshold_combined_frames).astype(int)

    config.add_field("thresholds", {
        "combined": float(optimal_threshold_combined_frames),
        "keypoints": float(optimal_threshold_keypoints_frames),
        "deep": float(optimal_threshold_deep_frames),
        "velocity": float(optimal_threshold_velocity_frames),
    })

    # Save results per frame
    for idx in range(len(final_scores_frames_velocity_all)):
        path = test_vel_files[idx]
        result_path = get_file_path(path, test_dirs.FRAME_LEVEL, FileExtension.NPY, subfolder_keep_count=1)

        np.save(
            result_path, {
                "prediction_velocity_frame": predictions_velocity_frames[idx],
                "prediction_deep_frame": predictions_deep_frames[idx],
                "prediction_keypoints_frame": predictions_keypoints_frames[idx],
                "prediction_combined_frame": predictions_combined_frames[idx],
                "object_predictions": object_predictions_per_frame[idx],
            }
        )

    config.close()

    create_roc_curve(path=os.path.join(test_dirs.RESULTS, "roc_curve_frame_velocity.png"), ground_truths=ground_truths, final_scores=final_scores_frames_velocity_all, micro_auc=micro_auc_velocity)
    create_roc_curve(path=os.path.join(test_dirs.RESULTS, "roc_curve_frame_deep.png"), ground_truths=ground_truths, final_scores=final_scores_frames_deep_all, micro_auc=micro_auc_deep)
    create_roc_curve(path=os.path.join(test_dirs.RESULTS, "roc_curve_frame_keypoints.png"), ground_truths=ground_truths, final_scores=final_scores_frames_keypoints_all, micro_auc=micro_auc_keypoints)
    create_roc_curve(path=os.path.join(test_dirs.RESULTS, "roc_curve_frame_combined.png"), ground_truths=ground_truths, final_scores=final_scores_frames_combined_all, micro_auc=micro_auc_combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech")
    parser.add_argument("--gmm", type=int, default=5)
    parser.add_argument("--knn", type=int, default=1)
    parser.add_argument("--sigma", type=int, default=7)
    parser.add_argument("--max-image-count-training", type=int, default=1000)  # This n samples populates the FAISS models and fits the GMM model
    parser.add_argument("--max-image-count-testing", type=int, default=1000)  # Actual evaluation on this n samples of the testing set
    args = parser.parse_args()

    evaluate(
        batch_size=50,
        gmm=args.gmm,
        knn=args.knn,
        sigma=args.sigma,
        dataset_name=args.dataset_name,
        max_image_count_training=args.max_image_count_training,
        max_image_count_testing=args.max_image_count_testing,
    )
