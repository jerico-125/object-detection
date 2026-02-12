# Object Detection Pipeline

End-to-end pipeline for creating privacy-preserving object detection datasets from video sources and training YOLOv8 models.

For detailed feature documentation, see [WORKFLOW_FEATURES.md](WORKFLOW_FEATURES.md).

## Project Structure

```
Object_Detection/
├── Pipeline/                  # 6-step modular workflow
│   ├── main.py                      # YOLO training workflow orchestrator
│   ├── extract.py                   # Video → frames (ffmpeg + histogram dedup)
│   ├── filter_images.py             # Interactive image curation (D/A/W/S keys)
│   ├── anonymize.py                 # Face & license plate blurring
│   ├── labeling.py                  # X-AnyLabeling launcher
│   ├── consolidate.py               # Dataset consolidation
│   ├── review_labels.py             # Label visualization & validation
│   ├── autolabel.py                 # YOLO auto-labeling inference
│   ├── inference.py                 # Video inference tool
│   ├── train.py                     # YOLOv8 training script
│   ├── model_utils.py               # Shared YOLO model selector
│   └── config.json                  # Pipeline configuration (self-documented)
├── Image_Sampling/            # Legacy standalone sampling/filtering tools
├── dataset_registry.json       # Tracks YOLO dataset version composition
├── YOLO_Training/             # Model training & format conversion
│   ├── convert_json_to_yolo.py      # JSON → YOLO format converter
│   ├── dataset.yaml                 # Dataset config template
│   ├── args.yaml                    # Auto-generated training record (NOT config input)
│   └── best.onnx                    # Exported trained model
├── WORKFLOW_FEATURES.md        # Detailed feature documentation
└── venv/                      # Python 3.12 virtual environment
```

## Data Flow

### Standard Workflow (image_labeling_workflow.py)
```
Video → [Extract Frames] → [Filter] → [Anonymize] → [Label] → [Consolidate] → [Review]
```

### YOLO Training Workflow (main.py)
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

Both workflows support: `--start-step N`, `--config file.json`, `--video path`, `--generate-config`.

```bash
cd Pipeline

# Standard workflow
python image_labeling_workflow.py

# YOLO training workflow
python main.py
python main.py --video video.mp4 --model best.pt --start-step 3
```

### Standalone Training

```bash
cd YOLO_Training
python convert_json_to_yolo.py --input_dir Dataset --output_dir Dataset_YOLO
python train.py --data dataset.yaml --epochs 100
```

## Configuration

All config keys are self-documented in `Pipeline/config.json` (see its `_documentation` section).

## Dataset Registry

`dataset_registry.json` at the project root tracks which source sets make up each YOLO dataset version. Each entry has `sources` (array of paths) and `notes` (free-text).

## UI Guidelines

### Color Scheme
- **Green** (`\033[92m`) — User input prompts (`input()` calls)
- **Red** (`\033[91m`) — Errors and warnings
- **Yellow** (`\033[93m`) — Configuration info the user should review/verify (parameters, paths, settings displayed before processing)
- **Cyan** (`\033[96m`) — Processing/status info (system telling the user what it's doing, summaries, results)

### Dividers
- **`=` * 60** — Major section headers (banners, summaries, step titles)
- **`-` * 60** — Sub-sections (config details, minor separators)

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
