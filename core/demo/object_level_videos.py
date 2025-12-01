import argparse
import os
import sys
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import cv2
import numpy as np
from tqdm.auto import tqdm

from core.config.paths import LoadPaths
from elements.config.load_config import Config
from elements.data.loaders.image.utils.imagecodecs import load_image
from elements.data.transforms.postprocess.visualize.basic import draw_text_with_box
from elements.data.utils import prune_lists_smallest_length, get_file_paths
from elements.enums.enums import FileExtension
from elements.save_results.result_saver import VideoResultSaver
from elements.visualize.display import Display


def create_videos_with_borders(dataset_name: str, max_image_count: int = -1, fps: int = 10):
    """Create a video visualizing frame-level predictions and ground truth."""
    test_dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="testing")

    frame_level_result_paths = get_file_paths(directory=test_dirs.FRAME_LEVEL, extensions=[FileExtension.NPY], recursive=True, sort_list=True, max_count=max_image_count)
    frames = get_file_paths(directory=test_dirs.FRAMES, extensions=[FileExtension.PNG, FileExtension.JPG, FileExtension.JPEG], recursive=True, sort_list=True, max_count=max_image_count)

    frames, frame_level_result_paths = prune_lists_smallest_length([frames, frame_level_result_paths])

    if not frames:
        print("No frames found. Exiting.")
        return

    height, width, _ = load_image(frames[0]).shape

    # Enable or disable visualization by changing the argument
    display_combined = Display(window_name="Combined score results")
    display_keypoints = Display(window_name="Keypoints score results")
    display_velocity = Display(window_name="Velocity score results")
    display_deep = Display(window_name="Deep features score results")

    output_video_path_velocity = os.path.join(test_dirs.RESULTS, f"{dataset_name}_velocity_object_level.mp4")
    output_video_path_deep = os.path.join(test_dirs.RESULTS, f"{dataset_name}_deep_object_level.mp4")
    output_video_path_combined = os.path.join(test_dirs.RESULTS, f"{dataset_name}_combined_object_level.mp4")
    output_video_path_keypoints = os.path.join(test_dirs.RESULTS, f"{dataset_name}_keypoints_object_level.mp4")

    video_saver_velocity = VideoResultSaver(out_file=output_video_path_velocity)
    video_saver_deep = VideoResultSaver(out_file=output_video_path_deep)
    video_saver_combined = VideoResultSaver(out_file=output_video_path_combined)
    video_saver_keypoints = VideoResultSaver(out_file=output_video_path_keypoints)

    video_saver_velocity.initiate_result_video(width=width, height=height, fps=fps)
    video_saver_deep.initiate_result_video(width=width, height=height, fps=fps)
    video_saver_combined.initiate_result_video(width=width, height=height, fps=fps)
    video_saver_keypoints.initiate_result_video(width=width, height=height, fps=fps)

    video_modes = [
        ("deep", video_saver_deep, display_deep),
        ("keypoints", video_saver_keypoints, display_keypoints),
        ("velocity", video_saver_velocity, display_velocity),
        ("combined", video_saver_combined, display_combined),
    ]

    config = Config()

    for frame_idx in tqdm(range(len(frame_level_result_paths)), desc="Processing frames"):
        frame_path = frames[frame_idx]
        result_path = frame_level_result_paths[frame_idx]

        # Load frame
        frame_original = load_image(frame_path)

        # Load object predictions
        result = np.load(result_path, allow_pickle=True).item()
        object_predictions_list = result.get("object_predictions", [])

        for mode, video_saver, display in video_modes:
            frame = deepcopy(frame_original)

            threshold = config.get("thresholds")[mode]

            # Loop through all objects in this frame
            for obj_pred in object_predictions_list:
                score = obj_pred.get(f"{mode}_score", 0.0)
                box = obj_pred["box"]
                matched_gt = obj_pred.get("matched_gt")

                is_predicted_anomaly = (score > threshold)

                # CASE 1: No detection and no GT → True Negative
                if box is None and matched_gt is None:
                    continue

                # CASE 2: GT anomaly but missed detection → False Negative
                elif box is not None and matched_gt == "missed":
                    # Draw the GT box that was missed
                    gt_box = box
                    color = (0, 0, 255)  # Red for missed anomaly
                    cv2.rectangle(frame, (int(gt_box.x1), int(gt_box.y1)), (int(gt_box.x2), int(gt_box.y2)), color, 2)
                    text = f"Missed GT | score={score:.3f}"
                    draw_text_with_box(frame, text, (int(gt_box.x1), max(int(gt_box.y1) - 10, 0)), color)

                # CASE 3: Prediction exists (matched or not)
                elif box is not None:
                    if matched_gt is None:
                        # False positive prediction
                        color = (0, 255, 255) if not is_predicted_anomaly else (0, 165, 255)  # Cyan or Orange
                        label = f"FP | score={score:.3f}"
                    else:
                        # True positive prediction
                        color = (255, 0, 0) if is_predicted_anomaly else (0, 255, 0)  # Red = anomaly, Green = normal
                        label = f"TP | score={score:.3f}"

                    # Draw the predicted box
                    cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
                    draw_text_with_box(frame, label, (int(box.x1), max(int(box.y1) - 10, 0)), color)

                # Append frame to video
                video_saver.append_image_to_video(image=frame)
                display.show_image(image=frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech", help='Name given to dataset')
    parser.add_argument("--max-image-count", type=int, default=1000, help='Limit amount of frames for processing')
    parser.add_argument("--fps", type=int, default=10, help='FPS of created video')
    args = parser.parse_args()

    create_videos_with_borders(dataset_name=args.dataset_name, max_image_count=args.max_image_count, fps=args.fps)
