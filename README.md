# KendamaCV

Real-time kendama trick recognition using MediaPipe Pose, HSV color tracking, and an LSTM classifier running in the browser via ONNX Runtime Web.

## Project Structure

```
kendama-cv/
├── index.html                   # App entry point
├── style.css                    # UI styles
├── app.js                       # Main orchestrator
├── pipeline/
│   ├── labels.js                # Trick definitions, constants
│   ├── pose.js                  # MediaPipe Pose wrapper + skeleton drawing
│   ├── detector.js              # YOLOv8 ONNX ken/tama detection + NMS
│   ├── tracker.js               # HSV color tracker (fallback)
│   ├── features.js              # Feature vector extraction + normalisation
│   └── lstm.js                  # ONNX LSTM inference + heuristic fallback
├── capture/
│   └── recorder.js              # Webcam + video file data collection
├── train/
│   ├── trainer.py               # Offline LSTM training → exports kendama.onnx
│   └── train_detector.py        # YOLOv8-nano fine-tuning → exports detector.onnx
├── data/
│   ├── dataset.json             # LSTM training data (from capture tool)
│   └── detector/
│       ├── dataset.yaml         # YOLO dataset config
│       ├── images/
│       │   ├── train/           # Training images
│       │   └── val/             # Validation images
│       └── labels/
│           ├── train/           # YOLO format .txt labels
│           └── val/
└── models/
    ├── kendama.onnx             # Trained LSTM classifier
    └── detector.onnx           # Trained YOLOv8 ken/tama detector
```

## Running Locally

Because the app uses ES modules and webcam access, it must be served over HTTP (not opened as a file).

```bash
# Python
python -m http.server 8080

# Node
npx serve .
```

Then open `http://localhost:8080`.

## Workflow

### 1. Calibrate (HSV fallback only)

If not using the detector, open the **Live** tab and use the color pickers or **Sample** buttons to set HSV ranges for your ken and tama. Click **Sample** then click directly on the object in the video feed. Skip this step if you load `detector.onnx`.

### 2. Collect detector training data

```bash
python train/train_detector.py scaffold
```

Add images to `data/detector/images/train` and `data/detector/images/val`. Label them with [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com) using two classes: `ken` (0) and `tama` (1). Aim for 200+ images per class across varied lighting, backgrounds, and angles.

### 3. Train the detector

```bash
pip install ultralytics
python train/train_detector.py train --data data/detector/dataset.yaml --out models/detector.onnx
```

### 4. Collect LSTM training data

Switch to the **Capture** tab. Select a trick label, choose webcam or video file, and record clips. Export `dataset.json` when done.

### 5. Train the LSTM

```bash
pip install torch numpy scikit-learn
python train/trainer.py --data data/dataset.json --out models/kendama.onnx
```

### 6. Load models & run inference

In the **Live** tab, load both `detector.onnx` and `kendama.onnx` using their respective file pickers. The detector runs first each frame, feeding bounding boxes and orientation into the LSTM feature pipeline.

> **Fallback behaviour:** Without `detector.onnx`, HSV color tracking is used (configure via the calibration panel). Without `kendama.onnx`, a rule-based heuristic classifier runs instead — labels shown in yellow.

## Feature Vector (113 dims per frame)

| Component | Dims | Description |
|---|---|---|
| Pose landmarks | 99 | 33 MediaPipe landmarks × (x, y, visibility) |
| Joint angles | 8 | Shoulders, elbows, wrists, hips |
| Ken orientation | 2 | Angle from vertical (norm), pixel length (norm) |
| Tama position | 4 | Relative X/Y, euclidean distance, angle to ken |

## Tricks

| ID | Name | Key motion |
|---|---|---|
| `spike` | Spike | Tama descends onto spike, ken vertical |
| `big_cup` | Big Cup | Tama catches in large cup |
| `lighthouse` | Lighthouse | Ken balances vertically on tama |
| `around_japan` | Around Japan | Tama orbits ken through 360° |
| `airplane` | Airplane | Tama swings horizontal to spike |
| `bird` | Bird | String taut balance, ken tilted |

## Tips for Good Training Data

- Record 15–30 clips per trick, ~3 seconds each
- Vary lighting, ken/tama colors, body position
- Include **failed attempts** as negative examples
- Keep the camera steady and at ~1m distance

## Pipeline In Depth

### Overview

```
Camera → Pose Estimation → Ken/Tama Tracking → Feature Extraction → LSTM → Trick Prediction
```

Each animation frame (~33ms at 30fps) flows through every stage in sequence. The final prediction is emitted once per sliding window (every 30 frames, ~1 second).

---

### Stage 1 — Camera

The browser `getUserMedia` API opens a webcam stream into a hidden `<video>` element. A `<canvas>` on top is what the user actually sees. The video element is never displayed directly — it acts purely as a pixel source for downstream stages.

---

### Stage 2 — Pose Estimation (`pipeline/pose.js`)

**Technology:** MediaPipe Pose (BlazePose), running fully in the browser via WASM.

Each frame, the raw video element is passed to `pose.send()`. MediaPipe returns 33 body landmarks, each with normalised coordinates `(x, y)` in `[0, 1]` relative to the frame, plus a `visibility` confidence score.

The landmarks used most heavily for kendama are:

- **Wrists** (15, 16) — where the player holds the ken or string
- **Elbows** (13, 14) and **Shoulders** (11, 12) — arm extension during throws
- **Hips** (23, 24) — body lean and crouch during catches

From these, 8 joint angles are computed using the dot-product formula between limb vectors:

```
angle(A, B, C) = acos( (A-B)·(C-B) / |A-B||C-B| )
```

These angles capture the shape of the throw/catch motion independent of where the player is standing in frame.

---

### Stage 3 — Ken & Tama Detection (`pipeline/detector.js` + `pipeline/tracker.js`)

Ken and tama localization uses two methods depending on what models are loaded. Both expose the same 6-element feature sub-vector so the rest of the pipeline is unaffected.

```
[ kenAngle_norm, kenLength_norm, tamaRelX, tamaRelY, tamaDist_norm, tamaAngle_norm ]
```

Tama position is always expressed relative to the ken centre — this relative geometry is what distinguishes most tricks (tama above vs below ken, close vs far, orbiting vs stationary).

---

#### Method A — YOLOv8 Object Detector (`detector.js`) — preferred

**Technology:** YOLOv8-nano fine-tuned on two classes (`ken`, `tama`), exported to ONNX, running in-browser via ONNX Runtime Web.

The detector is active when `detector.onnx` is loaded. Each frame goes through a full detection pipeline:

**Pre-processing**
The video frame is drawn to a 640×640 off-screen canvas and read into a `Float32Array` in CHW layout (channels-first), normalised to `[0, 1]`. This matches the input format YOLOv8 expects.

**Inference**
The tensor `[1, 3, 640, 640]` is passed to the ONNX session. YOLOv8-nano's output is shaped `[1, 6, 8400]` — 8400 anchor candidates, each with `(cx, cy, w, h, conf_ken, conf_tama)`.

**Post-processing**
1. Filter candidates below `CONF_THRESH` (0.35).
2. For each candidate, pick the highest-confidence class.
3. Run per-class **Non-Maximum Suppression** (NMS, IoU threshold 0.45) to eliminate duplicate boxes.
4. Take the top-1 detection per class.

**Ken orientation from bounding box**
YOLOv8 provides axis-aligned bounding boxes, not rotated ones. Ken orientation is derived from the box aspect ratio: a tall narrow box → ken is near-vertical (~90°); a wide short box → ken is near-horizontal (~0°). The longer dimension is used as the length estimate.

For more precise angle estimation in a future iteration, a segmentation head or keypoint model could replace this heuristic.

**Training the detector** — see `train/train_detector.py`:
```bash
# Create dataset folder structure
python train/train_detector.py scaffold

# Fine-tune from YOLOv8-nano pretrained weights
python train/train_detector.py train --data data/detector/dataset.yaml --out models/detector.onnx
```

Label images using LabelImg or Roboflow with two classes: `ken` (0) and `tama` (1). Aim for 200+ images per class across varied lighting, backgrounds, and ken/tama colors.

---

#### Method B — HSV Color Tracker (`tracker.js`) — fallback

Used automatically when no detector model is loaded. Labels are unaffected; the feature vector format is identical.

**How it works:**
1. Each frame is read into a pixel buffer via `getImageData`.
2. Every pixel is converted from RGB to HSV. HSV separates color identity (hue) from lighting conditions (value), making segmentation more robust under varying light than raw RGB.
3. Pixels within the calibrated HSV range for the ken are collected as a point cloud; same for the tama.
4. **Ken shape** — PCA line fitting on the ken point cloud. The covariance matrix's principal eigenvector gives the orientation axis; projecting points onto it gives length. Result: `{ cx, cy, angle, length }`.
5. **Tama shape** — centroid of the tama point cloud. Result: `{ cx, cy }`.

**Calibration:**
- **Color picker** — sets the HSV range symmetrically around the selected hex color.
- **Pixel sampling** — click Sample then click directly on the object in the video feed; the pixel's HSV becomes the range center.

---

### Stage 4 — Feature Extraction (`pipeline/features.js`)

All per-frame signals are concatenated into a single **113-dimensional feature vector**:

```
[ pose_landmarks (99) | joint_angles (8) | ken_tama (6) ] = 113 dims
```

**Online normalisation** is applied using Welford's one-pass algorithm — no pre-computed mean/std required. Each feature dimension is z-scored against its running mean and variance across all frames seen so far in the session. This makes the model robust to different players, lighting, and distances without retraining.

---

### Stage 5 — LSTM Classifier (`pipeline/lstm.js`)

**Technology:** ONNX Runtime Web — runs the trained PyTorch model directly in the browser via WebAssembly, with no server required.

A **circular buffer** of the last 30 feature vectors is maintained (the sliding window). Once the buffer is full, a `[1 × 30 × 113]` input tensor is built and passed to the ONNX session.

The model architecture (trained offline in `trainer.py`):
```
Input [batch, 30, 113]
  → LSTM (128 hidden, 2 layers, dropout 0.3)
  → Last timestep hidden state [batch, 128]
  → Dropout
  → Linear [128 → 6]
  → Softmax → class probabilities
```

The LSTM sees the full 1-second motion trajectory before committing to a prediction. This is what separates it from a per-frame classifier — tricks like Around Japan and Airplane require observing the arc of motion over time, not just a single pose snapshot.

Predictions are only surfaced when the top class confidence exceeds `CONF_THRESHOLD` (default 0.55), and a debounce of 45 frames prevents the same trick from being logged twice in rapid succession.

**Heuristic fallback** — when no `.onnx` model is loaded, a rule-based classifier runs instead using simple conditions on ken angle and tama position (e.g. `ken vertical AND tama above AND close → spike`). Output is shown in yellow to indicate it is not model-driven.

---

### Stage 6 — Output

Each prediction updates:
- **Trick name + confidence** in the prediction panel
- **Per-class confidence bars** (live, updated every frame)
- **Ken orientation indicator** — arrow rotates to match the detected ken angle
- **Tama position indicator** — dot moves to show tama position relative to ken centre
- **Trick history log** — debounced, timestamped log in the sidebar

---

## Dependencies

| Tool | Purpose |
|---|---|
| MediaPipe Pose (CDN) | Body landmark detection |
| ONNX Runtime Web (CDN) | In-browser LSTM + detector inference |
| Ultralytics YOLOv8 | Detector fine-tuning |
| PyTorch | LSTM offline training |
| scikit-learn | Train/val split, metrics |