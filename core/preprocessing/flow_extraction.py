import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join('third_party/NeuFlow_v2'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config.paths import LoadPaths
from elements.data.datatypes.inference_input import InferenceInput
from elements.data.transforms.postprocess.visualize.basic import flow_to_rgb
from elements.data.transforms.preprocess.image import Compose, ConvertRGB, ToFloat, Resize, HWC2CHW, ToTensor, ToCUDA, ToHalf, AddBatchDim, resize_image
from elements.data.utils import get_file_path
from elements.enums.enums import FileExtension
from elements.visualize.display import Display

from elements.data.loaders.processor import FrameProcessor
from elements.data.readers.multi_video import MultiVideoImageReader
from elements.data.samplers.video.last import LastFrameSampler
from elements.load_model.flow.neuflowv2 import NeuFlowV2


def extracting_flows(dataset_name: str, split_name: str = "training", max_image_count: int = -1):
    width_neuflow = 768
    height_neuflow = 432

    dirs = LoadPaths.get_dirs(dataset_name=dataset_name, split_name=split_name)

    neuflow = NeuFlowV2(weights_path=os.path.join("third_party", "NeuFlow_v2", "neuflow_sintel.pth"), image_width=width_neuflow, image_height=height_neuflow)
    reader = MultiVideoImageReader(dataset_path=dirs.FRAMES, max_image_count=max_image_count)

    input_transforms = Compose([
        ToFloat(),
        Resize(width=width_neuflow, height=height_neuflow),
        HWC2CHW(),
        ToTensor(),
        ToCUDA(),
        ToHalf(),
        AddBatchDim(),
    ])

    frame_processor = FrameProcessor(reader=reader, transforms=input_transforms, sampler=LastFrameSampler(sequence_length=1))

    display = Display()

    for idx, scs in frame_processor.frames():
        # prepare inference input
        inference_input = InferenceInput(sc=scs)

        # run prediction
        inference_result = neuflow.predict(x=inference_input)

        # resize flow output
        flow_resized = resize_image(inference_result.get("flow"), width=scs[0].org_width, height=scs[0].org_height)

        # save flow
        flow_path = get_file_path(file_path=scs[0].image_fpath, directory=dirs.FLOWS, extension=FileExtension.NPY, subfolder_keep_count=1)
        print(f"Saving flow to {flow_path}")
        np.save(flow_path, inference_result.get("flow"))

        # visualization
        flow_rgb = flow_to_rgb(flow=flow_resized)
        display.show_image(flow_rgb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shanghaitech", help='Name given to dataset')
    parser.add_argument("--split-name", type=str, default="all", choices=["training", "testing", "all"], help='Split of dataset to process')
    parser.add_argument("--max-image-count", type=int, default=1000, help='Limit amount of frames for processing')
    args = parser.parse_args()

    if args.split_name == "all":
        extracting_flows(dataset_name=args.dataset_name, split_name="training", max_image_count=args.max_image_count)
        extracting_flows(dataset_name=args.dataset_name, split_name="testing", max_image_count=args.max_image_count)
    else:
        extracting_flows(dataset_name=args.dataset_name, split_name=args.split_name, max_image_count=args.max_image_count)
