from typing import Any, List

from ultralytics import YOLO

from elements.data.datatypes.inference_input import InferenceInput
from elements.data.datatypes.inference_result import InferenceResult
from elements.load_model.model_base import BaseModel
from elements.processing.yolo import decode_yolo_boxes, decode_yolo_keypoints


class YoloPose(BaseModel):
    def __init__(self, weights_path: str, device: str = "cuda:0"):
        self.weights_path = weights_path
        self.device = device
        self.model = self.load_model()

    def predict(self, x: InferenceInput) -> InferenceResult:
        # Extract image tensor from SampleContainer
        if isinstance(x.sc, list):
            img = x.sc[0].image_data.get()
        else:
            img = x.sc.image_data.get()

        predictions = self.model.predict(img, conf=x.threshold, device=self.device, verbose=False)

        inference_result = InferenceResult(image=x.sc[0].org_image if isinstance(x.sc, list) else x.sc.org_image)

        if len(predictions) == 0:
            return inference_result

        r = predictions[0]

        boxes = decode_yolo_boxes(r=r, threshold=x.threshold)
        keypoints = decode_yolo_keypoints(r=r)

        # Use add() method for generalized storage
        inference_result.add("boxes", boxes)
        inference_result.add("keypoints", keypoints)

        return inference_result

    def load_model(self) -> Any:
        """
        Load the ultralytics YOLO model. We return the YOLO object and pass device into predict().
        """
        print(f"Loading YOLO weights from: {self.weights_path}")
        model = YOLO(self.weights_path)
        return model
