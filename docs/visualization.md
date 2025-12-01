## 2.1 create_results_videos.py – Frame-Level and Object-Level Video Generation

The project includes two visualization utilities that convert prediction files into annotated MP4 videos. These videos make it easier to understand how the model behaves over time, either at the level of whole frames or individual detected objects.

### Frame-Level Visualization

This mode generates four synchronized videos, one per scoring branch:

- Deep feature score
- Keypoint score
- Velocity score
- Combined score

For every frame, the script:

- Loads the raw RGB image.
- Colors the outer border according to the ground truth (normal or anomalous).
- Colors a dashed inner rectangle according to the model prediction.
- Draws small text labels (“Ground Truth”, “Prediction”) for clarity.
- Streams annotated frames to a preview window.
- Writes the result to MP4 files.

This provides a smooth, chronological view of how each feature modality contributes to anomaly detection.

Note: Changing the thresholds in the config does not change the used predictions for Frame-Level Visualization, as they are determined in the evaluation step. 

![frame-level-video.png](src/frame-level-video.png)

## Object-Level Visualization

This second utility focuses on detected objects within each frame. It produces one annotated video for each scoring branch.

### For each frame, it:

- Loads all per-object predictions and anomaly scores.
- Applies the branch-specific anomaly threshold.
- Draws bounding boxes for:
    - True Positives — detected anomalies
    - False Positives — predicted anomalies that aren’t real
    - False Negatives — missed anomalies
    - True Negatives (optional diagnostics)
- Color-codes each box (green/red/orange/cyan).
- Streams the result to a preview window and exports an MP4 file.

This view highlights exactly which objects trigger anomaly flags, making it easier to debug missing detections or extra noisy predictions.

Note: You can change the thresholds in the config for this step if the calculated values seem off.

![object-level-video.png](src/object-level-video.png)