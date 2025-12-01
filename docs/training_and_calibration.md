# 1.4 score_calibration.py – Score Calibration for Deep, Keypoint, and Velocity Features

This script learns how “large” or “small” anomaly-relevant scores naturally are across the entire training set. Since deep embeddings, keypoints, and velocity histograms all live in different numerical universes, calibration provides a shared language for the anomaly pipeline.

## Purpose

The calibration stage computes global normalization ranges for:

- Deep features (CLIP embeddings)
- Keypoint features
- Velocity histograms

These ranges are later used to transform raw anomaly distances into normalized scores for inference. Without calibration, each feature type would contribute values on incompatible scales.

### Pipeline

#### 1. Loads object-level feature files

From the dataset’s preprocessed directories:

- Object-level deep feature files (DEEP_FEATURES/)
- Object-level keypoint files (KEYPOINTS/)
- Object-level velocity files (VELOCITIES/)

Each entry corresponds to one frame’s collection of object descriptors.

#### 2. Learns velocity statistics using a Gaussian Mixture Model

Velocity histograms (8-bin representations of motion direction/magnitude) are collected across the training set.
The script fits a GMM to capture the shape of normal motion behavior.
From the training samples, it extracts:

- min_vel
- max_vel

These define the numerical window for velocity-based distances during inference.

#### 3. Trains two FAISS KNN models (leave-one-video-out)

To avoid a model cheating by comparing objects to themselves, the script trains:

- One KNN index for deep features
- One KNN index for keypoints

Both are trained in a leave-one-video-out fashion:

- For each video, a dedicated KNN model is trained on all the others
- Distances for the held-out video are then computed cleanly
- This greatly reduces bias in score scaling

Distances use k = 2, with the second neighbor used as the anomaly distance.

The script tracks:

- min_deep, max_deep
- min_keypoints, max_keypoints

These become the normalization bounds.

#### 4. Writes results back into global configuration

All computed statistics are inserted into a calibration block inside config.yml so the entire inference pipeline can reuse them:

```yaml
calibration:
  min_deep: ...
  max_deep: ...
  min_keypoints: ...
  max_keypoints: ...
  min_vel: ...
  max_vel: ...
```

During testing, raw distances from all modalities are linearly mapped into the calibrated range.

## Usage

Run calibration:

```bash
python ../core/preprocessing/score_calibration.py \
    --dataset-name shanghaitech \
    --max-image-count 4000
```

| Argument            | Default          | Description                                       |
|---------------------|------------------|---------------------------------------------------|
| `--dataset-name`    | `"shanghaitech"` | Dataset entry from `config.yml`.                  |
| `--max-image-count` | `4000`           | Max number of feature/velocity files to consider. |
