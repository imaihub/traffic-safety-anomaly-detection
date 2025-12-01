import argparse
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config.paths import LoadPaths

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter

from elements.data.datatypes.inference_input import InferenceInput
from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.annotation.boundingbox import LoadBoundingBoxes
from elements.data.loaders.annotation.keypoints import LoadKeypoints
from elements.data.loaders.image.utils.imagecodecs import load_image
from elements.data.transforms.preprocess.image import Compose, ToFloat, Resize, HWC2CHW, ToTensor, ToCUDA, ToHalf, AddBatchDim, resize_image
from elements.data.transforms.preprocess.normalize.custom import NormalizeCustom
from elements.data.transforms.preprocess.normalize.enums import NormalizeValues
from elements.data.transforms.preprocess.normalize.normal import Normalize
from elements.data.utils import prune_lists_smallest_length, get_file_paths, get_file_path
from elements.enums.enums import FileExtension
from elements.load_model.extractors.clip import CLIP


def extract_velocity(flow: np.ndarray, magnitude: float, orientation: int, orientations: int = 8, motion_threshold: float = 0.):
    orientation *= (180 / np.pi)

    cy, cx = flow.shape[:2]

    orientation_histogram = np.zeros(orientations)
    subsample = np.index_exp[cy // 2:cy:cy, cx // 2:cx:cx]
    for i in range(orientations):
        temp_ori = np.where(orientation < 360 / orientations * (i + 1), orientation, -1)

        temp_ori = np.where(orientation >= 360 / orientations * i, temp_ori, -1)

        cond2 = (temp_ori > -1) * (magnitude >= motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))

        orientation_histogram[i] = temp_filt[subsample]

    return orientation_histogram


def extract_features(dataset_name: str, split_name: str = "training", max_image_count: int = -1):
    dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name=split_name)
    frame_paths = get_file_paths(directory=dirs.FRAMES, extensions=[FileExtension.PNG, FileExtension.JPG, FileExtension.JPEG], recursive=True, sort_list=True, max_count=max_image_count)
    flow_paths = get_file_paths(directory=dirs.FLOWS, extensions=[FileExtension.NPY], recursive=True, sort_list=True, max_count=max_image_count)
    boxes_paths = get_file_paths(directory=dirs.BOXES, extensions=[FileExtension.XML], recursive=True, sort_list=True, max_count=max_image_count)

    frame_paths, flow_paths, boxes_paths = prune_lists_smallest_length([frame_paths, flow_paths, boxes_paths])

    input_transforms = Compose([
        ToFloat(),
        Resize(224, 224),
        HWC2CHW(),
        ToTensor(),
        ToCUDA(),
        ToHalf(),
        AddBatchDim(),
        Normalize(),
        NormalizeCustom(mean=NormalizeValues.IMAGE_NET_MEAN, std=NormalizeValues.IMAGE_NET_STD),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIP(device=device)

    load_bboxes = LoadBoundingBoxes()
    load_keypoints = LoadKeypoints()

    for i_frame, (frame_path, flow_path, boxes_path) in enumerate(zip(frame_paths, flow_paths, boxes_paths)):
        print(f"Processing frame {frame_path} with {len(frame_paths) - i_frame} images to go")
        frame = load_image(frame_path)
        frame_boxes = load_bboxes(boxes_path)
        frame_keypoints = load_keypoints(boxes_path)
        frame_flow = np.load(flow_path)

        deep_features_per_frame = defaultdict(list)
        velocities_features_per_frame = defaultdict(list)
        for i_box, box in enumerate(frame_boxes):
            frame_crop = frame[int(box.y1):int(box.y2), int(box.x1):int(box.x2), :]
            frame_crop = input_transforms(frame_crop)

            frame_flow = resize_image(img=frame_flow, width=frame.shape[1], height=frame.shape[0])
            flow_crop = frame_flow[int(box.y1):int(box.y2), int(box.x1):int(box.x2), :]

            sc = SampleContainer()
            sc.image_data = frame_crop
            sc.org_height, sc.org_width = frame_crop.shape[:2]
            sc.org_image = frame

            # Deep features from CLIP
            inference_input = InferenceInput(sc=sc)
            inference_result = model.predict(x=inference_input)

            df = inference_result.get("features")
            deep_features_per_frame["features"].append(df)
            deep_features_per_frame["boxes"].append([box.y1, box.x1, box.y2, box.x2])

            keypoints_in_box = []
            for kp in frame_keypoints:
                if box.y1 <= kp.y <= box.y2 and box.x1 <= kp.x <= box.x2:
                    kp.x = (kp.x - box.x1) / box.width
                    kp.y = (kp.y - box.y1) / box.height

                    keypoints_in_box.append((kp.y, kp.x))

            if len(keypoints_in_box) >= 17:
                keypoints_in_box = keypoints_in_box[:17]
                deep_features_per_frame["keypoints"].append(np.array(keypoints_in_box).reshape(1, 34))
            elif len(keypoints_in_box) < 17:
                deep_features_per_frame["keypoints"].append(np.zeros((1, 34)))

            # Velocity from flow
            img_flow = np.transpose(flow_crop, [1, 2, 0]).astype(np.float32)
            magnitude, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])

            mag = np.sqrt(img_flow[..., 0] ** 2 + img_flow[..., 1] ** 2)
            mag = mag / box.height
            velocity_cur = extract_velocity(flow=img_flow, magnitude=mag, orientation=ang, orientations=8)
            train_velocity = velocity_cur[None]
            velocities_features_per_frame["velocities"].append(train_velocity)
            velocities_features_per_frame["boxes"].append([box.y1, box.x1, box.y2, box.x2])

        deep_features_path = get_file_path(file_path=frame_path, directory=dirs.DEEP_FEATURES, extension=FileExtension.NPY, subfolder_keep_count=1)
        velocity_path = get_file_path(file_path=frame_path, directory=dirs.VELOCITIES, extension=FileExtension.NPY, subfolder_keep_count=1)
        keypoints_path = get_file_path(file_path=frame_path, directory=dirs.KEYPOINTS, extension=FileExtension.NPY, subfolder_keep_count=1)

        if len(frame_boxes) == 0:
            np.save(deep_features_path, {
                "features": np.zeros((0, 512)),
                "boxes": [],
            })
            np.save(velocity_path, {
                "velocities": np.zeros((0, 8)),
                "boxes": [],
            })
            np.save(keypoints_path, {
                "keypoints": np.zeros((0, 34)),
                "boxes": [],
            })
            continue

        velocities_features_per_frame["velocities"] = np.concatenate(velocities_features_per_frame["velocities"], axis=0)
        np.save(velocity_path, velocities_features_per_frame)

        deep_features_per_frame["features"] = np.concatenate(deep_features_per_frame["features"], axis=0)
        np.save(deep_features_path, deep_features_per_frame)

        deep_features_per_frame["keypoints"] = np.concatenate(deep_features_per_frame["keypoints"], axis=0)
        np.save(keypoints_path, deep_features_per_frame["keypoints"].astype(np.float32))

    print(f"Completed processing Train split")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech", help='Name given to dataset')
    parser.add_argument("--split-name", type=str, default="all", choices=["training", "testing", "all"], help='Split of dataset to process')
    parser.add_argument("--max-image-count", type=int, default=10000, help='Limit amount of frames for processing')
    args = parser.parse_args()

    if args.split_name == "all":
        extract_features(dataset_name=args.dataset_name, split_name="training", max_image_count=args.max_image_count)
        extract_features(dataset_name=args.dataset_name, split_name="testing", max_image_count=args.max_image_count)
    else:
        extract_features(dataset_name=args.dataset_name, split_name=args.split_name, max_image_count=args.max_image_count)
