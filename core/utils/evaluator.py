import os

import numpy as np

from elements.data.datatypes.annotations.boundingbox import BoundingBox, BoundingBoxProcessor
from elements.data.transforms.utils.annotation import class_mask_to_bounding_box


class ObjectLevelEvaluator:
    def __init__(self, gt_mask_paths, velocity_scores_objects, deep_scores_objects, keypoints_scores_objects, combined_scores_objects, predicted_boxes):
        self.gt_mask_paths = gt_mask_paths
        self.velocity_scores_objects = velocity_scores_objects
        self.keypoints_scores_objects = keypoints_scores_objects
        self.deep_scores_objects = deep_scores_objects
        self.combined_scores_objects = combined_scores_objects
        self.predicted_boxes = predicted_boxes

        # Results
        self.object_predictions_per_frame = []
        self.velocity_scores_flat = []
        self.keypoints_scores_flat = []
        self.combined_scores_flat = []
        self.deep_scores_flat = []

    def run(self):
        """Main entry point for evaluation."""
        for directory in self.velocity_scores_objects.keys():
            self._process_directory(directory)
        return self._results()

    # ---- Private helpers ----
    def _process_directory(self, directory):
        try:
            gt_mask_file = next(path for path in self.gt_mask_paths if os.path.basename(path).split(".")[0] == os.path.basename(directory))
        except StopIteration:
            print(f"Could not find GT for {directory}")
            return

        gt_mask_video = np.load(gt_mask_file)
        velocity_scores = self.velocity_scores_objects[directory]
        deep_scores = self.deep_scores_objects[directory]
        keypoints_scores = self.deep_scores_objects[directory]
        combined_scores = self.combined_scores_objects[directory]
        pred_boxes = self.predicted_boxes[directory]

        for i, (v_scores_frame, d_scores_frame, k_scores_frame, c_scores_frame, boxes) in enumerate(zip(velocity_scores, deep_scores, keypoints_scores, combined_scores, pred_boxes)):
            current_mask = gt_mask_video[i]
            gt_boxes = class_mask_to_bounding_box(current_mask)

            if len(boxes) == 0:
                frame_predictions = self._handle_no_predictions(i, gt_boxes)
            else:
                bounding_boxes = []
                for b in boxes:
                    box = BoundingBox()
                    box.set_minmax_yx(ymin=b[0], xmin=b[1], ymax=b[2], xmax=b[3])
                    bounding_boxes.append(box)
                if len(gt_boxes) == 0:
                    frame_predictions = self._handle_no_gt(i, bounding_boxes, v_scores_frame, d_scores_frame, k_scores_frame, c_scores_frame)
                else:
                    frame_predictions = self._handle_predictions_and_gt(i, bounding_boxes, gt_boxes, v_scores_frame, d_scores_frame, k_scores_frame, c_scores_frame)

            self.object_predictions_per_frame.append(frame_predictions)

    def _handle_no_predictions(self, frame_index, gt_boxes):
        frame_predictions = []
        if len(gt_boxes) == 0:
            # TN: no GT, no prediction
            self._append_results(0, 0.0, 0.0, 0.0, 0.0)
            frame_predictions.append(self._make_pred(None, frame_index, 0, 0.0, 0.0, 0.0, 0.0, None))
        else:
            # FN: missed GT boxes
            for gt_box in gt_boxes:
                self._append_results(1, 0.0, 0.0, 0.0, 0.0)
                frame_predictions.append(self._make_pred(gt_box, frame_index, 1, 0.0, 0.0, 0.0, 0.0, "missed"))
        return frame_predictions

    def _handle_no_gt(self, frame_index, bounding_boxes, v_scores, d_scores, k_scores, c_scores):
        frame_predictions = []
        for box, vs, ds, ks, cs in zip(bounding_boxes, v_scores, d_scores, k_scores, c_scores):
            self._append_results(0, vs, ds, ks, cs)
            frame_predictions.append(self._make_pred(box, frame_index, 0, vs, ds, ks, cs, None))
        return frame_predictions

    def _handle_predictions_and_gt(self, frame_index, bounding_boxes, gt_boxes, v_scores, d_scores, k_scores, c_scores):
        bbox_processor = BoundingBoxProcessor(boxes=bounding_boxes)
        matches = bbox_processor.match_bboxes(boxes2=gt_boxes, min_iou=0.01)
        frame_predictions = []
        for box, vs, ds, ks, cs, gt_index in zip(bounding_boxes, v_scores, d_scores, k_scores, c_scores, matches):
            label = 1 if gt_index != -1 else 0
            matched_gt = gt_boxes[gt_index] if gt_index != -1 else None
            self._append_results(label, vs, ds, ks, cs)
            frame_predictions.append(self._make_pred(box, frame_index, label, vs, ds, ks, cs, matched_gt))
        return frame_predictions

    # ---- Utility ----
    def _append_results(self, label, vs, ds, ks, cs):
        self.velocity_scores_flat.append(float(vs))
        self.deep_scores_flat.append(float(ds))
        self.keypoints_scores_flat.append(float(ks))
        self.combined_scores_flat.append(float(cs))

    def _make_pred(self, box, frame_index, label, vs, ds, ks, cs, matched_gt):
        return {"box": box, "frame_index": frame_index, "is_anomaly": label, "velocity_score": float(vs), "deep_score": float(ds), "keypoints_score": float(ks), "combined_score": float(cs), "matched_gt": matched_gt}

    def _results(self):
        return {
            "object_predictions_per_frame": self.object_predictions_per_frame,
            "velocity_scores_flat": np.array(self.velocity_scores_flat),
            "deep_scores_flat": np.array(self.deep_scores_flat),
            "combined_scores_flat": np.array(self.combined_scores_flat),
            "keypoints_scores_flat": np.array(self.keypoints_scores_flat),
        }
