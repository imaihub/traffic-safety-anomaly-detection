import argparse
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.join('third_party/NeuFlow_v2'))

from core.config.paths import LoadPaths
from core.preprocessing.feature_extraction import extract_velocity
from elements.config.load_config import Config
from elements.data.datatypes.annotations.boundingbox import BoundingBoxProcessor, delete_overlapped_bboxes, get_foreground_bboxes
from elements.data.datatypes.annotations.keypoint import resize_keypoints
from elements.data.datatypes.inference_input import InferenceInput
from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.processor import FrameProcessor
from elements.data.readers.multi_video import MultiVideoImageReader
from elements.data.samplers.video.last import LastFrameSampler
from elements.data.transforms.postprocess.visualize.basic import draw_text_with_box
from elements.data.transforms.preprocess.image import ToFloat, Compose, Resize, HWC2CHW, ToTensor, ToCUDA, AddBatchDim, ToHalf, resize_image
from elements.data.transforms.preprocess.normalize.custom import NormalizeCustom
from elements.data.transforms.preprocess.normalize.enums import NormalizeValues
from elements.data.transforms.preprocess.normalize.normal import Normalize
from elements.data.utils import get_file_paths
from elements.enums.enums import FileExtension
from elements.load_model.extractors.clip import CLIP
from elements.load_model.extractors.faiss_model import FaissModel
from elements.load_model.extractors.gmm import GMMModel
from elements.load_model.flow.neuflowv2 import NeuFlowV2
from elements.load_model.pose.yolo import YoloPose
from elements.visualize.display import Display

def inference(dataset_name: str, gmm: int, knn: int, max_image_count_training: int = -1, max_image_count_testing: int = -1, batch_size: int = 50, detector_threshold: float = 0.1, area_threshold: int = 1000, binary_threshold: float = 0.1, gauss_mask_size: int = 5, cover_threshold: float = 0.1):
    """Video anomaly detection, all inference steps combined"""
    # Loading file lists, reading config and initializing readers
    test_dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="testing")
    train_dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name="training")

    train_vel_files = get_file_paths(train_dirs.VELOCITIES, [FileExtension.NPY], recursive=True, max_count=max_image_count_training)
    train_deep_files = get_file_paths(train_dirs.DEEP_FEATURES, [FileExtension.NPY], recursive=True, max_count=max_image_count_training)
    train_keypoints_files = get_file_paths(train_dirs.KEYPOINTS, [FileExtension.NPY], recursive=True, max_count=max_image_count_training)

    # Enable or disable visualization by changing the argument
    display_combined = Display(window_name="Combined score results", enabled=True)
    display_keypoints = Display(window_name="Keypoints score results", enabled=True)
    display_velocity = Display(window_name="Velocity score results", enabled=False)
    display_deep = Display(window_name="Deep features score results", enabled=True)

    config = Config()
    threshold_combined = config.get("thresholds")["combined"]
    threshold_velocity = config.get("thresholds")["velocity"]
    threshold_keypoints = config.get("thresholds")["keypoints"]
    threshold_deep = config.get("thresholds")["deep"]

    reader = MultiVideoImageReader(dataset_path=test_dirs.FRAMES, max_image_count=max_image_count_testing)

    transforms_detection = Compose([
        ToFloat(),
        Resize(width=640, height=640),
        HWC2CHW(),
        ToTensor(),
        ToCUDA(),
        AddBatchDim(),
        Normalize(),
    ])
    transforms_flow = Compose([
        ToFloat(),
        Resize(width=768, height=432),
        HWC2CHW(),
        ToTensor(),
        ToCUDA(),
        ToHalf(),
        AddBatchDim(),
    ])
    transforms_feature = Compose([
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

    frame_processor_detection = FrameProcessor(reader=reader, transforms=transforms_detection)
    frame_processor_flow = FrameProcessor(reader=reader, transforms=transforms_flow, sampler=LastFrameSampler(sequence_length=1))

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

    # Initialize Detection, Flow and Clip models
    detection_model = YoloPose(weights_path='yolov8x-pose-p6.pt')
    neuflow = NeuFlowV2(weights_path=os.path.join("third_party", "NeuFlow_v2", "neuflow_sintel.pth"), image_width=768, image_height=432)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model = CLIP(device=device)

    for (i, scs_boxes), (_, scs_flows) in zip(frame_processor_detection.frames(), frame_processor_flow.frames()):
        visualization_image_velocity = deepcopy(scs_flows[0].org_image.get())
        visualization_image_keypoints = deepcopy(scs_flows[0].org_image.get())
        visualization_image_deep = deepcopy(scs_flows[0].org_image.get())
        visualization_image_combined = deepcopy(scs_flows[0].org_image.get())

        # Object Detection
        sc_boxes = scs_boxes[0]

        inference_input = InferenceInput(sc=sc_boxes, threshold=detector_threshold)
        inference_result = detection_model.predict(inference_input)

        boxes = inference_result.get("boxes") or []
        boxes = [box for box in boxes if box.area > area_threshold]

        scale_x = sc_boxes.org_width / 640
        scale_y = sc_boxes.org_height / 640

        bbox_processor = BoundingBoxProcessor(boxes=boxes)
        bbox_processor.resize_boxes(scale_x=scale_x, scale_y=scale_y)
        boxes = bbox_processor.get_boxes()

        keypoints = inference_result.get("keypoints") or []
        keypoints = resize_keypoints(keypoints=keypoints, scale_x=scale_x, scale_y=scale_y)

        obj_bboxes_after_overlap_removal = delete_overlapped_bboxes(bboxes=boxes, cover_threshold=cover_threshold)
        foreground_bboxes = get_foreground_bboxes(img_batch=np.asarray([sc_boxes.org_image.data, sc_boxes.org_image.data]), bboxes=obj_bboxes_after_overlap_removal, area_threshold=area_threshold, binary_threshold=binary_threshold, gauss_mask_size=gauss_mask_size)

        boxes = obj_bboxes_after_overlap_removal + foreground_bboxes

        # Flow
        inference_input_flow = InferenceInput(sc=scs_flows)
        flow_result = neuflow.predict(x=inference_input_flow)
        flow_resized = resize_image(flow_result.get("flow"), width=scs_flows[0].org_width, height=scs_flows[0].org_height)

        # Features per crop
        for box in boxes:
            processing_image = sc_boxes.org_image.get()
            frame_crop = processing_image[int(box.y1):int(box.y2), int(box.x1):int(box.x2), :]
            frame_crop = transforms_feature(frame_crop)

            frame_flow = resize_image(img=flow_resized, width=processing_image.shape[1], height=processing_image.shape[0])
            flow_crop = frame_flow[int(box.y1):int(box.y2), int(box.x1):int(box.x2), :]

            sc = SampleContainer()
            sc.image_data = frame_crop
            sc.org_height, sc.org_width = frame_crop.shape[:2]
            sc.org_image = sc_boxes.org_image

            # Deep features from CLIP
            inference_input = InferenceInput(sc=sc)
            inference_result = clip_model.predict(x=inference_input)

            features = inference_result.get("features")

            keypoints_in_box = []
            for kp in keypoints:
                if box.y1 <= kp.y <= box.y2 and box.x1 <= kp.x <= box.x2:
                    kp.x = (kp.x - box.x1) / box.width
                    kp.y = (kp.y - box.y1) / box.height

                    keypoints_in_box.append((kp.y, kp.x))

            if len(keypoints_in_box) >= 17:
                keypoints_in_box = keypoints_in_box[:17]
                keypoints_in_box = np.array(keypoints_in_box).reshape(1, 34)
            elif len(keypoints_in_box) < 17:
                keypoints_in_box = np.zeros((1, 34))

            # Velocity from flow
            img_flow = np.transpose(flow_crop, [1, 2, 0]).astype(np.float32)
            magnitude, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])

            mag = np.sqrt(img_flow[..., 0]**2 + img_flow[..., 1]**2)
            mag = mag / box.height
            velocity_cur = extract_velocity(flow=img_flow, magnitude=mag, orientation=ang, orientations=8)
            velocity = velocity_cur[None]

            v_score_objects = gmm_model.get_score(data=velocity, normalize=True)
            d_score_objects = faiss_model_features.get_score(data=features, knn=knn, normalize=True)
            k_score_objects = faiss_model_keypoints.get_score(data=keypoints_in_box, knn=knn, normalize=True)

            combined_score_objects = (v_score_objects + d_score_objects + k_score_objects) / 3

            # Drawing results per mode
            # Combined
            is_predicted_anomaly_combined = (combined_score_objects > threshold_combined)
            color = (0, 0, 255) if is_predicted_anomaly_combined else (255, 0, 0)
            cv2.rectangle(visualization_image_combined, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
            text = f"Combined score={float(combined_score_objects[0]):.3f}"
            draw_text_with_box(visualization_image_combined, text, (int(box.x1), max(int(box.y1) - 10, 0)), color)

            # Velocity
            is_predicted_anomaly_velocity = (v_score_objects > threshold_velocity)
            color = (0, 0, 255) if is_predicted_anomaly_velocity else (255, 0, 0)
            cv2.rectangle(visualization_image_velocity, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
            text = f"Velocity score={float(v_score_objects[0]):.3f}"
            draw_text_with_box(visualization_image_velocity, text, (int(box.x1), max(int(box.y1) - 10, 0)), color)

            # Deep
            is_predicted_anomaly_deep = (d_score_objects > threshold_deep)
            color = (0, 0, 255) if is_predicted_anomaly_deep else (255, 0, 0)
            cv2.rectangle(visualization_image_deep, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
            text = f"Deep features score={float(d_score_objects[0]):.3f}"
            draw_text_with_box(visualization_image_deep, text, (int(box.x1), max(int(box.y1) - 10, 0)), color)

            # Keypoints
            is_predicted_anomaly_keypoints = (d_score_objects > threshold_keypoints)
            color = (0, 0, 255) if is_predicted_anomaly_keypoints else (255, 0, 0)
            cv2.rectangle(visualization_image_keypoints, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
            text = f"Pose score={float(d_score_objects[0]):.3f}"
            draw_text_with_box(visualization_image_keypoints, text, (int(box.x1), max(int(box.y1) - 10, 0)), color)

        display_combined.show_image(image=visualization_image_combined)
        display_deep.show_image(image=visualization_image_deep)
        display_velocity.show_image(image=visualization_image_velocity)
        display_keypoints.show_image(image=visualization_image_keypoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech", help='Name given to dataset')
    parser.add_argument("--max-image-count-training", type=int, default=1000) # This n samples populates the FAISS models and fits the GMM model
    parser.add_argument("--max-image-count-testing", type=int, default=1000) # Actual evaluation on this n samples of the testing set
    parser.add_argument("--gmm", type=int, default=5)
    parser.add_argument("--knn", type=int, default=1)
    args = parser.parse_args()

    inference(dataset_name=args.dataset_name, max_image_count_training=args.max_image_count_training, max_image_count_testing=args.max_image_count_testing, gmm=args.gmm, knn=args.knn)
