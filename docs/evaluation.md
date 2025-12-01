# 1.5 evaluation.py – Frame- and Object-Level Anomaly Detection Evaluation

This module runs the complete evaluation pipeline on a dataset’s testing split.
It uses the calibrated deep, keypoint, and velocity features to produce:

- Frame-level anomaly scores
- Object-level anomaly scores
- Temporally smoothed scores
- Micro-AUC metrics
- Optimal ROC thresholds
- Final per-frame predictions
- ROC curve visualizations

This is the script that ties together training features, calibration parameters, model inference, and result aggregation.

## What it does

### 1. Load training and testing feature files

The script retrieves:

- deep_features/
- keypoints/
- velocities/

for both the training and testing splits.
Lists are pruned to identical lengths to keep deep/keypoint/velocity features perfectly aligned.

Optional max_count limits restrict how many samples are used.

### 2. Load calibration parameters

Reads the precomputed calibration block from the global config:

- min_deep, max_deep
- min_keypoints, max_keypoints
- min_vel, max_vel

These values normalize model outputs so that scores across all modalities share a common scale.

### 3. Train inference models (using training data only)

The evaluation constructs three inference models:

- GMMModel (velocity)
  Learns the probability density of normal motion via a Gaussian Mixture Model.

- FaissModel (deep)
  kNN-based anomaly scoring over CLIP embeddings.

- FaissModel (keypoints)
  kNN-based scoring over pose/keypoint descriptors.

Each model uses the calibration ranges to produce normalized distances.

### 4. Compute anomaly scores on testing data

For each frame, the script computes:

- Velocity anomaly score
- Deep-feature anomaly score
- Keypoint anomaly score
- Combined score = mean of the three modalities

Object-level bounding boxes and their corresponding anomaly scores are also extracted.

All outputs are grouped per video.

### 5. Temporal smoothing

To reduce frame-level noise, each score sequence is smoothed with a Gaussian kernel (sigma):

- Smoothed velocity
- Smoothed deep
- Smoothed keypoints
- Smoothed combined

### 6. Compute Micro-AUC metrics

Flattened ground truth and predicted scores are used to compute:

- Velocity AUC
- Deep AUC
- Keypoints AUC
- Combined AUC

### 7. Object-level evaluation

Uses ObjectLevelEvaluator to match predictions and ground truths, and populate the score lists for missed and non-existent predictions.

### 8. Determine optimal thresholds

For each modality, evaluation builds an ROC curve and selects a threshold using Youden’s J-index: (Note: these thresholds can be manually adjusted in the code as sometimes the calculated threshold is off)

```text
threshold = argmax(TPR – FPR)
```

Thresholds are added to the global configuration under:

```text
thresholds:
velocity: ...
deep: ...
keypoints: ...
combined: ...
```

### 9. Save final per-frame prediction files

For each frame, the script stores a .npy file containing:

- prediction_velocity_frame
- prediction_deep_frame
- prediction_keypoints_frame
- prediction_combined_frame
- object_predictions

These files feed directly into the video-generation tools.

### 10. Save ROC curve images

Four ROC curve images are placed under the results/ directory:

- roc_curve_frame_velocity.png
- roc_curve_frame_deep.png
- roc_curve_frame_keypoints.png
- roc_curve_frame_combined.png

### Usage

Example:

```bash
python ../core/evaluation/evaluate_anomaly_detection.py \
    --dataset-name shanghaitech \
    --gmm 5 \
    --knn 1 \
    --sigma 7 \
    --max-image-count-training 1000 \
    --max-image-count-testing 1000
```

| Argument                     | Default        | Description                            |
|------------------------------|----------------|----------------------------------------|
| `--dataset-name`             | `shanghaitech` | Dataset name from `config.yml`.        |
| `--gmm`                      | `5`            | Number of Gaussian components for GMM. |
| `--knn`                      | `1`            | k for FAISS nearest-neighbor scoring.  |
| `--sigma`                    | `7`            | Gaussian smoothing kernel size.        |
| `--max-image-count-training` | `1000`         | Limit training samples.                |
| `--max-image-count-testing`  | `1000`         | Limit testing samples.                 |

