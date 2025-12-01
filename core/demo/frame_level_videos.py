import argparse
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
from tqdm.auto import tqdm

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.config.paths import LoadPaths
from elements.data.loaders.image.utils.imagecodecs import load_image
from elements.data.transforms.postprocess.visualize.basic import draw_dashed_rectangle, draw_text_with_box
from elements.data.utils import prune_lists_smallest_length, get_file_paths
from elements.enums.enums import FileExtension
from elements.save_results.result_saver import VideoResultSaver
from elements.visualize.display import Display


def create_videos_with_borders(dataset_name: str, max_image_count: int = -1, fps: int = 10):
    """Create a video visualizing frame-level predictions and ground truth."""
    test_dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="testing")

    frame_level_result_paths = get_file_paths(directory=test_dirs.FRAME_LEVEL, extensions=[FileExtension.NPY], recursive=True, sort_list=True, max_count=max_image_count)
    frames = get_file_paths(directory=test_dirs.FRAMES, extensions=[FileExtension.PNG, FileExtension.JPG, FileExtension.JPEG], recursive=True, sort_list=True, max_count=max_image_count)

    gt_files = get_file_paths(test_dirs.FRAME_MASK_GT, [FileExtension.NPY], recursive=True)
    all_gt = [np.load(f, allow_pickle=True).astype(np.int32).flatten() for f in gt_files]
    ground_truths = np.concatenate(all_gt, 0)

    frames, frame_level_result_paths, ground_truths = prune_lists_smallest_length([frames, frame_level_result_paths, ground_truths])

    if not frames:
        print("No frames found. Exiting.")
        return

    height, width, _ = load_image(frames[0]).shape

    # Enable or disable visualization by changing the argument
    display_combined = Display(window_name="Combined score results")
    display_keypoints = Display(window_name="Keypoints score results")
    display_velocity = Display(window_name="Velocity score results")
    display_deep = Display(window_name="Deep features score results")

    output_video_path_velocity = os.path.join(test_dirs.RESULTS, f"{dataset_name}_velocity_frame_level.mp4")
    output_video_path_deep = os.path.join(test_dirs.RESULTS, f"{dataset_name}_deep_frame_level.mp4")
    output_video_path_combined = os.path.join(test_dirs.RESULTS, f"{dataset_name}_combined_frame_level.mp4")
    output_video_path_keypoints = os.path.join(test_dirs.RESULTS, f"{dataset_name}_keypoints_frame_level.mp4")

    video_saver_velocity = VideoResultSaver(out_file=output_video_path_velocity)
    video_saver_deep = VideoResultSaver(out_file=output_video_path_deep)
    video_saver_combined = VideoResultSaver(out_file=output_video_path_combined)
    video_saver_keypoints = VideoResultSaver(out_file=output_video_path_keypoints)

    video_saver_velocity.initiate_result_video(width=width, height=height, fps=fps)
    video_saver_deep.initiate_result_video(width=width, height=height, fps=fps)
    video_saver_combined.initiate_result_video(width=width, height=height, fps=fps)
    video_saver_keypoints.initiate_result_video(width=width, height=height, fps=fps)

    # Precompute common values
    border_thickness = 2
    gap = 8
    inner_start = border_thickness + gap
    inner_end_w = width - border_thickness - gap - 1
    inner_end_h = height - border_thickness - gap - 1
    padding_top = 40

    for frame_idx in tqdm(range(len(ground_truths) - 1), desc="Processing frames"):
        frame_path = frames[frame_idx]
        result_path = frame_level_result_paths[frame_idx]

        frame_original = load_image(frame_path)
        result = np.load(result_path, allow_pickle=True).item()  # `.item()` if saved as dict

        for video_saver, display, prediction in [
            (video_saver_deep, display_deep, result["prediction_deep_frame"]),
            (video_saver_keypoints, display_keypoints, result["prediction_keypoints_frame"]),
            (video_saver_velocity, display_velocity, result["prediction_velocity_frame"]),
            (video_saver_combined, display_combined, result["prediction_combined_frame"]),
        ]:
            frame = deepcopy(frame_original)

            annotation = ground_truths[frame_idx]

            gt_color = (0, 255, 0) if annotation == 0 else (0, 0, 255)
            pred_color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

            # Draw rectangles
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), gt_color, border_thickness)
            draw_dashed_rectangle(frame, (inner_start, inner_start), (inner_end_w, inner_end_h), pred_color, border_thickness)
            draw_text_with_box(frame, "Ground Truth", (width - 100, padding_top), gt_color, is_dashed=False)
            draw_text_with_box(frame, "Prediction", (width - 100, padding_top + 25), pred_color, is_dashed=True)

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
