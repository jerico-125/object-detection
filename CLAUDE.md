# Object Detection Pipeline

End-to-end pipeline for creating privacy-preserving object detection datasets from video sources and training YOLOv8 models.

## Project Structure

```
Object_Detection/
├── Pipeline/                  # 6-step modular workflow
│   ├── image_labeling_workflow.py   # Main orchestrator
│   ├── extract_frames.py            # Video → frames (ffmpeg + histogram dedup)
│   ├── filter_images.py             # Interactive image curation (D/A/W/S keys)
│   ├── anonymize.py                 # Face & license plate blurring
│   ├── labeling.py                  # X-AnyLabeling launcher
│   ├── consolidate.py               # Dataset consolidation
│   ├── review_labels.py             # Label visualization & validation
│   ├── yolo_autolabel.py            # YOLO auto-labeling inference
│   └── workflow_config.json         # Pipeline configuration
├── Image_Sampling/            # Legacy standalone sampling/filtering tools
├── YOLO_Training/             # Model training & format conversion
│   ├── train_yolo.py                # YOLOv8 training script
│   ├── convert_json_to_yolo.py      # JSON → YOLO format converter
│   ├── dataset.yaml                 # Dataset config template
│   ├── args.yaml                    # Training hyperparameters
│   └── best.onnx                    # Exported trained model
└── venv/                      # Python 3.12 virtual environment
```

## Data Flow

```
Video → [Extract Frames] → [Filter] → [Anonymize] → [Label] → [Consolidate] → [Review] → [Convert to YOLO] → [Train YOLOv8] → ONNX model
```

## Setup

Requires: Python 3.12, ffmpeg/ffprobe, OpenCV, NumPy, Ultralytics, Pillow.
Optional: `understand-ai/anonymizer` (Step 3), X-AnyLabeling (Step 4).

```bash
source venv/bin/activate
```

## Running the Pipeline

```bash
cd Pipeline

# Interactive menu (select steps)
python image_labeling_workflow.py

# Start from a specific step
python image_labeling_workflow.py --start-step 3

# Custom config
python image_labeling_workflow.py --config custom_config.json

# Specify input video
python image_labeling_workflow.py --video /path/to/video.mp4

# Generate config template
python image_labeling_workflow.py --generate-config
```

## Training

```bash
cd YOLO_Training

# Convert JSON annotations to YOLO format
python convert_json_to_yolo.py --input_dir Dataset --output_dir Dataset_YOLO

# Train YOLOv8
python train_yolo.py --data dataset.yaml --epochs 100
```

## Annotation Formats

Supports JSON (LabelMe/X-AnyLabeling), YOLO TXT, and Pascal VOC XML.

## Key Configuration (workflow_config.json)

- `frame_threshold`: Histogram similarity for dedup (default 0.85)
- `target_fps`: Frame extraction rate (default 3.0)
- `blur_threshold`: Laplacian variance minimum (default 100.0)
- `face_threshold` / `plate_threshold`: Anonymization detection thresholds (default 0.3)
- `label_format`: Annotation format — `json`, `txt`, or `xml`

## UI Guidelines

- **Warnings must be colored red** — All warning messages displayed to the user must use red text (e.g., ANSI escape code `\033[91m` or equivalent).

## Logging Policy

All changes made to this project must be documented in [LOG.md](LOG.md). Writing to LOG.md does not require user permission — do it automatically after every change.

Each entry must include:
- Date and time
- Explicit list of which code files were modified (by filename)
- Bullet-point explanation of what was changed and why
