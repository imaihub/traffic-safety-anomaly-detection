# 1. Preprocessing Pipelines

The preprocessing stage converts raw frames into the intermediate representations the anomaly system needs: bounding boxes, motion fields, and object-level features.

## 1.1 bboxes_extraction.py - Bounding Box and Pose Extraction

This script detects objects, extracts poses, and produces structured annotations for every frame. It relies on YOLOv8 Pose and supplements detections with simple foreground–motion cues to catch moving objects that might slip past the model.

These annotations become the spine of the rest of the pipeline—velocity extraction, deep feature encoding, and object-level anomaly scoring.

### Pipeline

- Runs YOLOv8 Pose to extract bounding boxes + keypoints
- Filters detections based on:
    - minimum area
    - confidence threshold
    - overlapping bounding boxes (via IoU filtering)
- Adds motion-based pseudo-detections using:
    - frame differencing
    - Gaussian smoothing
- Rescales all detections back to original resolution
- Saves annotations in CVAT XML format (one file per frame)
- Optional visualizer for debugging
- Works with training and testing splits

### Usage

To extract bounding boxes:

```bash
python ../core/preprocessing/bboxes_extraction.py --dataset-name shanghaitech --split-name training --max-image-count 5000
```

Extract for both training + testing:

```bash
python ../core/preprocessing/bboxes_extraction.py --split-name all
```

Enable visualization:

```bash
python ../core/preprocessing/bboxes_extraction.py --visualize
```

| Argument               | Default          | Description                                  |
|------------------------|------------------|----------------------------------------------|
| `--dataset-name`       | `"shanghaitech"` | Dataset entry in `config.yml`.               |
| `--split-name`         | `"all"`          | One of: `training`, `testing`, `all`.        |
| `--max-image-count`    | `5000`           | Max frames to process per split.             |
| `--visualize`          | `False`          | Show frames with bounding boxes + keypoints. |
| `--detector-threshold` | `0.1`            | YOLO confidence threshold.                   |
| `--cover-threshold`    | `0.1`            | IoU cutoff for dropping overlapping boxes.   |
| `--binary-threshold`   | `0.1`            | Pixel-diff threshold for motion detector.    |
| `--gauss-mask-size`    | `5`              | Gaussian blur kernel size.                   |
| `--area-threshold`     | `1000`           | Minimum area required for a box.             |

## 1.2 flows_extraction.py – Optical Flow Extraction

This script computes dense optical flow between consecutive frames using NeuFlowV2 (Sintel pretrained).

Each predicted flow field is resized to the original frame dimensions and saved as a .npy tensor.

These motion fields are later distilled into velocity histograms—one of the core signals used by the anomaly modules.

#### Pipeline

- Computes dense optical flow with NeuFlowV2
- Outputs one flow file per frame: shape H × W × 2
- Rescales flow fields to match full-resolution images
- Optional visualization using RGB flow colorization
- Efficient sequential frame loading via FrameProcessor and MultiVideoImageReader

Usage
Extract flow maps for training:

```bash
python ../core/preprocessing/flow_extraction.py --dataset-name shanghaitech --split-name training --max-image-count 1000
```

Flow files are placed under:

```text
.../flows/<clip>/<frame>.npy
```

### 1.3 features_extraction.py — Deep Features & Motion Features

This script builds the final, object-level representation used by the anomaly engine.
For each detected object in every frame, it extracts three essential ingredients:

- Deep appearance features via CLIP
- Pose features (normalized keypoints)
- Velocity histograms distilled from optical flow

Outputs are stored as .npy files mirroring the frame directory structure.

All outputs mirror the frame folder hierarchy and are stored as .npy files.

### Pipeline

#### 1. Load:

- RGB frame
- Dense optical flow (flow.npy)
- XML bounding boxes
- Keypoints

For each bounding box:

- Crops the image and flow fields
- Runs a preprocessing pipeline (resize → normalize → tensor → GPU)
- Encodes the crop with CLIP → 512-D deep feature vector
- Extracts keypoints that fall inside the box
- Computes an 8-bin velocity orientation histogram using flow magnitude & direction
- Stores all outputs grouped by object

Frames with zero detections produce empty arrays to maintain alignment.

### Output Structure

#### Velocity Features

Saved to: velocities/<frame>.npy

{
"velocities": (N, 8),
"boxes":      (N, 4)
}

#### Deep Features (CLIP)

Saved to: deep_features/<frame>.npy

{
"features":  (N, 512),
"boxes":     (N, 4),
"keypoints": (N, 34)
}

#### Keypoints Only

Saved to: keypoints/<frame>.npy

Shape: (N, 34)
Where N = number of objects in the frame.

##### Velocity Features — Details

Velocity for each object is computed as:

- Normalize flow magnitude by bounding-box height
- Convert orientation to degrees
- Bin orientation into 8 uniform bins
- Apply uniform filtering for stability
- Produce a compact (1 × 8) descriptor

These histograms give each object a small pulse of motion energy, sliced by direction.

### Usage

```bash
python ../core/preprocessing/feature_extraction.py \
    --dataset-name shanghaitech \
    --split-name training \
    --max-image-count 2000
```

| Argument            | Default          | Description                                      |
|---------------------|------------------|--------------------------------------------------|
| `--dataset-name`    | `"shanghaitech"` | Dataset configuration key in `config.yml`.       |
| `--split-name`      | `"all"`          | One of `training`, `testing`, `all`.             |
| `--max-image-count` | `10000`          | Maximum number of frames to process (per split). |

