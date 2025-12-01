## Dataset preparation

Before the system can extract motion, learn normality, or bless you with anomaly-tinted videos, your dataset needs to follow a precise layout. Think of this as the scaffolding on which all later stages hang.

## 1. Dataset Layout

Each video is represented as a folder of sequential frames. Clip names are arbitrary but must be unique.

```text
/path/to/dataset/
    00_001/
        001.jpg
        002.jpg
        ...
    00_002/
        001.jpg
        002.jpg
        ...
```

A training set requires at least two clips, because the density estimators need multiple independent samples to form their “sense of normalcy.”

## 2. Configuring Paths

Update the dataset paths in core/config/config.yml

This tells every script where to find input frames, where to put intermediate features, and where to store results.

```text
shanghaitech:
  training_root: /path/to/training/
  testing_root: /path/to/testing/

  results: /path/to/results/

  training_frames: /path/to/training/frames
  testing_frames: /path/to/testing/frames

  pixel_mask_gt: /path/to/test_pixel_mask
  frame_mask_gt: /path/to/test_frame_mask
```

### What belongs where?

    - *_frames → raw frames used by preprocessing.
    - *_root → where preprocessing outputs (bboxes, flows, features, etc.) will accumulate.
    - results → final anomaly curves, logs, and visualizations.
    - *_mask_gt → ground-truth annotations.

## 3. Ground Truth Format (Optional)

Ground truth is only needed for evaluation. Inference runs without it.

#### Pixel-level Ground Truth (pixel_mask_gt)

    - Stored as: .npy
    - Shape: (n, H, W)
    - Values:
        - 1 → anomalous
        - 0 → normal
    - n corresponds to the frame index within the clip.

#### Frame-level Ground Truth (frame_mask_gt)

    - Stored as: .npy
    - Shape: (n,)
    - Values:
        - 1 → anomalous frame
        - 0 → normal frame

These are consumed by the evaluator to compute ROC curves, AUC, and thresholded performance.

## 4. Directory Roles

A small field guide for all the paths referenced in config.yml:

| Path                                 | Purpose                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
| `training_root`                      | Stores preprocessing outputs for the training split.                    |
| `testing_root`                       | Same as above, but for testing.                                         |
| `results`                            | Final plots, graphs, videos, logs, thresholds.                          |
| `training_frames` / `testing_frames` | Directories containing the raw frames for each clip.                    |
| `pixel_mask_gt` / `frame_mask_gt`    | Used only during evaluation to compute per-pixel and per-frame metrics. |
