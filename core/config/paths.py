import os
from dataclasses import dataclass

from elements.config.yaml_parser import get_yaml_dict
from elements.data.utils import ensure_dir_exists


@dataclass
class Dirs:
    ROOT: str
    FRAMES: str
    BOXES: str
    FLOWS: str
    DEEP_FEATURES: str
    VELOCITIES: str
    KEYPOINTS: str
    FRAME_LEVEL: str
    PIXEL_MASK_GT: str
    FRAME_MASK_GT: str
    RESULTS: str


class LoadPaths:
    @staticmethod
    def get_dirs(dataset_name: str, split_name: str = "training") -> Dirs:
        config_parser = get_yaml_dict(os.path.join("core", "config", "config.yml"))["paths"][dataset_name]
        root_dir = config_parser[f"{split_name}_root"]
        frames_dir = config_parser[f"{split_name}_frames"]
        boxes_dir = os.path.join(root_dir, "boxes")
        flows_dir = os.path.join(root_dir, "flows")
        deep_features_dir = os.path.join(root_dir, "deep_features")
        velocities_dir = os.path.join(root_dir, "velocities")
        keypoints_dir = os.path.join(root_dir, "keypoints")

        frame_mask_gt_dir = config_parser["frame_mask_gt"]
        pixel_mask_gt_dir = config_parser["pixel_mask_gt"]
        results_dir = config_parser["results"]

        frame_level_results_dir = os.path.join(results_dir, "frame_level")

        ensure_dir_exists(dirs=[boxes_dir, flows_dir, deep_features_dir, velocities_dir, frames_dir, results_dir])

        dirs = Dirs(ROOT=root_dir, FRAMES=frames_dir, BOXES=boxes_dir, FLOWS=flows_dir, DEEP_FEATURES=deep_features_dir, VELOCITIES=velocities_dir, FRAME_LEVEL=frame_level_results_dir, KEYPOINTS=keypoints_dir, PIXEL_MASK_GT=pixel_mask_gt_dir, FRAME_MASK_GT=frame_mask_gt_dir, RESULTS=results_dir)
        return dirs
