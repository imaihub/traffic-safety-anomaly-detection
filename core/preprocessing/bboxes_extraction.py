import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config.paths import LoadPaths
from elements.data.utils import get_file_path

from elements.data.datatypes.annotations.boundingbox import delete_overlapped_bboxes, get_foreground_bboxes, BoundingBoxProcessor
from elements.data.datatypes.annotations.keypoint import resize_keypoints
from elements.data.datatypes.inference_input import InferenceInput
from elements.data.loaders.annotation.cvat import CVATAnnotations
from elements.data.loaders.processor import FrameProcessor
from elements.data.readers.multi_video import MultiVideoImageReader
from elements.data.transforms.postprocess.visualize.basic import visualize
from elements.data.transforms.preprocess.image import Compose, ToFloat, Resize, HWC2CHW, ToTensor, ToCUDA, AddBatchDim
from elements.data.transforms.preprocess.normalize.normal import Normalize
from elements.enums.enums import FileExtension
from elements.load_model.pose.yolo import YoloPose

from elements.visualize.display import Display


def extract_boxes(dataset_name: str, split_name: str = "training", max_image_count: int = -1, detector_threshold: float = 0.1, area_threshold: int = 1000, binary_threshold: float = 0.1, gauss_mask_size: int = 5, cover_threshold: float = 0.1, display_enabled: bool = False):
    dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name=split_name)

    model = YoloPose(weights_path='yolov8x-pose-p6.pt')

    width, height = 640, 640
    reader = MultiVideoImageReader(dataset_path=dirs.FRAMES, max_image_count=max_image_count)

    transforms = Compose([
        ToFloat(),
        Resize(width=width, height=height),
        HWC2CHW(),
        ToTensor(),
        ToCUDA(),
        AddBatchDim(),
        Normalize(),
    ])
    frame_processor = FrameProcessor(reader=reader, transforms=transforms)

    display = Display(enabled=display_enabled)

    for idx, scs in frame_processor.frames():
        sc = scs[0]

        annotation_file_path = get_file_path(file_path=sc.image_fpath, directory=dirs.BOXES, extension=FileExtension.XML, subfolder_keep_count=1)
        with CVATAnnotations(save_path=annotation_file_path, task_name='Box and keypoint annotations', img_count=1, bbox_class_names=["object"], keypoint_class_names=["object"]) as saver:
            # Wrap SampleContainer(s) in InferenceInput object
            inference_input = InferenceInput(sc=sc, threshold=detector_threshold)

            # Run model
            inference_result = model.predict(inference_input)

            # Filter boxes
            boxes = inference_result.get("boxes") or []
            boxes = [box for box in boxes if box.area > area_threshold]

            # Scale for visualization
            scale_x = sc.org_width / width
            scale_y = sc.org_height / height

            bbox_processor = BoundingBoxProcessor(boxes=boxes)
            bbox_processor.resize_boxes(scale_x=scale_x, scale_y=scale_y)
            boxes = bbox_processor.get_boxes()

            keypoints = inference_result.get("keypoints") or []
            keypoints = resize_keypoints(keypoints=keypoints, scale_x=scale_x, scale_y=scale_y)

            # Remove overlaps and detect motion
            obj_bboxes_after_overlap_removal = delete_overlapped_bboxes(bboxes=boxes, cover_threshold=cover_threshold)
            foreground_bboxes = get_foreground_bboxes(img_batch=np.asarray([sc.org_image.data, sc.org_image.data]), bboxes=obj_bboxes_after_overlap_removal, area_threshold=area_threshold, binary_threshold=binary_threshold, gauss_mask_size=gauss_mask_size)

            cur_bboxes = obj_bboxes_after_overlap_removal + foreground_bboxes

            # Visualization
            vis_img = visualize(
                img=sc.org_image.data,
                boxes=cur_bboxes,
                keypoints=keypoints,
            )
            display.show_image(vis_img)

            # Save annotations
            saver.save_bboxes(bboxes=cur_bboxes, image_fname=sc.image_fpath, image_id=idx, image_width=sc.org_width, image_height=sc.org_height, clip_coordinates=True)
            saver.save_keypoints(keypoints=keypoints, image_fname=sc.image_fpath, image_id=idx, image_width=sc.org_width, image_height=sc.org_height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech", help='Name given to dataset')
    parser.add_argument("--split-name", type=str, default="all", choices=["training", "testing", "all"], help='Split of dataset to process')
    parser.add_argument("--max-image-count", type=int, default=5000, help='Limit amount of frames for processing')
    parser.add_argument("--visualize", action="store_true", help="Visualize processed frame including boxes and keypoints")

    parser.add_argument("--detector-threshold", type=float, default=0.1, help="Minimum confidence threshold for the pose or object detector; lower values include more detections, higher values make detection stricter.")
    parser.add_argument("--cover-threshold", type=float, default=0.1, help="IoU threshold for removing overlapping bounding boxes; boxes with overlap higher than this are treated as duplicates and removed.")
    parser.add_argument("--binary-threshold", type=float, default=0.1, help="Pixel intensity difference threshold used for motion detection between consecutive frames; higher values make motion detection less sensitive.")
    parser.add_argument("--gauss-mask-size", type=int, default=5, help="Kernel size for Gaussian blurring applied before motion detection; larger values smooth more noise but can blur fine details.")
    parser.add_argument("--area-threshold", type=int, default=1000, help="Minimum area (in pixels) for detected motion regions to be considered valid; smaller regions are ignored as noise.")

    args = parser.parse_args()

    if args.split_name == "all":
        extract_boxes(
            dataset_name=args.dataset_name,
            split_name="training",
            max_image_count=args.max_image_count,
            detector_threshold=args.detector_threshold,
            area_threshold=args.area_threshold,
            binary_threshold=args.binary_threshold,
            gauss_mask_size=args.gauss_mask_size,
            cover_threshold=args.cover_threshold,
            display_enabled=args.visualize
        )
        extract_boxes(
            dataset_name=args.dataset_name,
            split_name="testing",
            max_image_count=args.max_image_count,
            detector_threshold=args.detector_threshold,
            area_threshold=args.area_threshold,
            binary_threshold=args.binary_threshold,
            gauss_mask_size=args.gauss_mask_size,
            cover_threshold=args.cover_threshold,
            display_enabled=args.visualize
        )
    else:
        extract_boxes(
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            max_image_count=args.max_image_count,
            detector_threshold=args.detector_threshold,
            area_threshold=args.area_threshold,
            binary_threshold=args.binary_threshold,
            gauss_mask_size=args.gauss_mask_size,
            cover_threshold=args.cover_threshold,
            display_enabled=args.visualize
        )
