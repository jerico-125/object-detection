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
│   ├── autolabel.py                 # YOLO auto-labeling inference
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

### Standard Workflow (image_labeling_workflow.py)
```
Video → [Extract Frames] → [Filter] → [Anonymize] → [Label] → [Consolidate] → [Review]
```

### YOLO Training Workflow (yolo_training_workflow.py)
```
Video → [Extract Frames] → [YOLO Auto-Label] → [Anonymize] → [Review/Correct] → [Consolidate & Convert to YOLO] → [Train YOLOv8] → ONNX model
```

## Setup

Requires: Python 3.12, ffmpeg/ffprobe, OpenCV, NumPy, Ultralytics, Pillow.
Optional: `understand-ai/anonymizer` (Step 3), X-AnyLabeling (Step 4).

```bash
source venv/bin/activate
```

## Running the Pipeline

### Standard Workflow (Manual Labeling)

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

### YOLO Training Workflow (with Auto-Labeling)

```bash
cd Pipeline

# Interactive menu (select steps)
python yolo_training_workflow.py

# Start from a specific step (e.g., training only)
python yolo_training_workflow.py --start-step 6

# Specify video and model
python yolo_training_workflow.py --video /path/to/video.mp4 --model /path/to/best.pt

# Generate config template
python yolo_training_workflow.py --generate-config
```

**Steps:**
1. Extract image frames from video
2. YOLO auto-label (delete images with no detections)
3. Anonymize faces and license plates
4. Review/correct labels (X-AnyLabeling)
5. Consolidate & convert to YOLO format
6. Train YOLO model

## Training (Standalone)

If using the standard workflow or training separately:

```bash
cd YOLO_Training

# Convert JSON annotations to YOLO format
python convert_json_to_yolo.py --input_dir Dataset --output_dir Dataset_YOLO

# Train YOLOv8
python train_yolo.py --data dataset.yaml --epochs 100

# Export to ONNX (optional, also prompted after training)
python -c "from ultralytics import YOLO; YOLO('runs/train/weights/best.pt').export(format='onnx')"
```

**Note:** The YOLO training workflow (Step 6) performs training automatically. Standalone training is only needed when using the standard workflow or training outside the pipeline.

## Annotation Formats

Supports JSON (LabelMe/X-AnyLabeling), YOLO TXT, and Pascal VOC XML.

## Configuration Files

### workflow_config.json (Pipeline Configuration)

Located in `Pipeline/workflow_config.json`. Controls frame extraction, filtering, anonymization, labeling, and consolidation steps.

#### Frame Extraction & Clustering
- `frame_threshold`: Histogram similarity for dedup (default 0.85)
- `target_fps`: Frame extraction rate (default 3.0)
- `blur_threshold`: Laplacian variance minimum (default 100.0)

#### Anonymization
- `face_threshold` / `plate_threshold`: Anonymization detection thresholds (default 0.3)

#### Labeling & Consolidation
- `label_format`: Annotation format — `json`, `txt`, or `xml`

#### YOLO Auto-Labeling
- `yolo_model_path`: Path to trained YOLO model (.pt or .onnx)
- `autolabel_confidence`: Detection confidence threshold (default 0.25)
- `autolabel_delete_unlabeled`: Remove images with no detections (default true)
- `autolabel_output_dir`: Directory where labeled images and labels are copied to (default `./autolabeled`)
- `yolo_train_ratio`: Train/val split ratio (default 0.8)

### YOLO Training Configuration

YOLO training configuration is stored in **two locations** depending on how training is invoked:

#### 1. Integrated Workflow Training (yolo_training_workflow.py Step 6)

Training parameters can be configured in `workflow_config.json`:
- `train_model`: Model to use — `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc. (default `yolov8n.pt`)
- `train_epochs`: Number of training epochs (default 100)
- `train_batch`: Batch size (default 16)
- `train_imgsz`: Input image size (default 640)
- `train_device`: CUDA device (e.g., `0`, `0,1`, or `cpu`; empty = auto)
- `train_workers`: Number of dataloader workers (default 8)
- `train_project`: Output directory for training results (default `./runs`)
- `train_name`: Experiment name (default auto-generated timestamp)
- `train_resume`: Resume from last checkpoint (default false)
- `train_pretrained`: Use pretrained weights (default true)
- `train_optimizer`: Optimizer to use — `auto`, `SGD`, `Adam`, `AdamW` (default `auto`)
- `train_lr0`: Initial learning rate (default 0.01)
- `train_patience`: Early stopping patience (default 50)
- `train_cache`: Cache images for faster training (default false)
- `train_amp`: Use Automatic Mixed Precision (default true)
- `train_augment`: Enable data augmentation (default true)

**Note:** If a parameter is not found in `workflow_config.json`, the workflow falls back to defaults defined in `yolo_training_workflow.py` (lines 115-132 in `DEFAULT_CONFIG`).

#### 2. Standalone Training (train_yolo.py)

When running `train_yolo.py` directly, configuration comes from:
- **Command-line arguments** (see `python train_yolo.py --help`)
- **YOLO_Training/args.yaml** — Auto-generated file containing the last training run's parameters (NOT used as input, only as a record)
- **YOLO_Training/dataset.yaml** — Dataset configuration (paths, class names)

The `args.yaml` file is **auto-generated by Ultralytics** after training and serves as a record of what parameters were used. It is NOT used to configure new training runs.

## UI Guidelines

- **Warnings must be colored red** — All warning messages displayed to the user must use red text (e.g., ANSI escape code `\033[91m` or equivalent).

## Logging Policy

All changes made to this project must be documented in [LOG.md](LOG.md). Writing to LOG.md does not require user permission — do it automatically after every change.

Each entry must include:
- Date
- Explicit list of which code files were modified (by filename)
- Bullet-point explanation of what was changed and why

**IMPORTANT:**
- **Insertion point:** Always add new log entries directly below the `# Change Log` heading and `---` separator — above all existing date sections. If today's date section already exists at the top, append the new bullet points to that section (below its `## YYYY-MM-DD` heading) rather than creating a duplicate date heading.
- **Git commit separators:** After committing to git, add a commit separator **above** the log entries that were included in that commit. Format: `**Git commit <short-hash>**...` followed by `---`. The separator goes between the `# Change Log` / `---` heading and the `## date` section, so it is clear which entries belong to which commit.
- Never modify past log entries. Historical logs must remain unchanged to maintain an accurate project history.
- **DO NOT FORGET to record changes to LOG.md immediately after making any code modifications.** This includes edits, file renames, deletions, and new files. Always update LOG.md before committing to git.
