# YOLO Training Workflow - Complete Feature Documentation

This document provides a comprehensive breakdown of features for each step in the YOLO Training Pipeline.

---

## Table of Contents

- [Step 1: Extract Frames](#step-1-extract-frames)
- [Step 2: YOLO Auto-Label](#step-2-yolo-auto-label)
- [Step 3: Anonymize](#step-3-anonymize)
- [Step 4: Review/Correct Labels](#step-4-reviewcorrect-labels)
- [Step 5: Consolidate + YOLO Conversion](#step-5-consolidate--yolo-conversion)
- [Step 6: Train YOLO Model](#step-6-train-yolo-model)
- [Alternative Step 6: Review Labels Viewer](#alternative-step-6-review-labels-viewer)
- [General Features](#general-features)

---

## Step 1: Extract Frames

**File:** `extract_frames.py`
**Purpose:** Extract unique, sharp frames from videos or cluster existing images.

### Input Options

- **Single video file** → Extract frames with deduplication
- **Directory of videos** → Batch process all videos
- **Directory of images** → Apply clustering to existing images

### Deduplication Methods

#### Sequential Clustering (Default - Fast)
- **Complexity:** O(n × window_size)
- **How it works:** Compares each frame only to nearby neighbors
- **Best for:** Video frames (consecutive frames are similar)
- **Window size:** Configurable (default: 10 frames)
- **Algorithm:** Union-Find with chained similarity

#### DBSCAN Clustering (Thorough)
- **Complexity:** O(n²)
- **How it works:** Compares all frames to each other
- **Best for:** Mixed/shuffled images from multiple sources
- **Parameters:**
  - `eps` (distance threshold, auto-derived from similarity threshold)
  - `min_samples` (default: 2)

**User Prompt:** When processing images, asks which method to use (1=Sequential, 2=DBSCAN)

### Quality Filtering

#### Blur Detection
- **Method:** Laplacian variance
- **Threshold:** 100.0 (configurable)
- **Higher values** = sharper images required
- **Action:** Blurry frames skipped or moved to deleted folder

#### Histogram Similarity
- **Method:** HSV histogram correlation
- **Threshold:** 0.85 (configurable, 0-1 scale)
- **Higher values** = stricter deduplication
- **Bins:** 32 (configurable)
- **Action:** Keeps sharpest frame from similar groups

### Performance Optimizations

- **FFmpeg streaming:** No temporary files written
- **Parallel processing:** Multiprocessing pool for image directories
  - Max workers: min(CPU_count, 8)
  - Parallel histogram + blur computation
- **Batch statistics:** Timing breakdown shows:
  - Parallel processing time (load + blur + histogram)
  - Clustering algorithm time
  - File copying time
  - Other overhead

### Frame Control

- **Target FPS:** Extract at specific frame rate (default: 3.0 fps)
- **Max frames:** Optional limit on total frames extracted
- **Frame prefix:** Custom naming for output files (default: "frame")
- **Frame interval:** Auto-calculated from video FPS and target FPS

### Output

- **Location:** `./extracted_frames/<video_name>/`
- **Deleted files:** `./deleted/clustering/`
- **Statistics:**
  - Total frames processed
  - Blurry frames skipped
  - Similar frames skipped
  - Unique frames saved
  - Processing time

### Configuration Keys

```json
{
  "video_path": "",
  "extracted_frames_dir": "./extracted_frames",
  "frame_threshold": 0.85,
  "target_fps": 3.0,
  "blur_threshold": 100.0,
  "clustering": true,
  "clustering_method": null,  // null = prompt user
  "clustering_eps": null,  // auto-derived as 1.0 - frame_threshold
  "clustering_min_samples": 2,
  "clustering_window_size": 10,
  "max_frames": null,
  "frame_prefix": "frame",
  "histogram_bins": 32,
  "clustering_deleted_dir": "./deleted/clustering"
}
```

---

## Step 2: YOLO Auto-Label

**File:** `autolabel.py`
**Purpose:** Automatically label images using a trained YOLO model and remove unlabeled images.

### Model Support

- **PyTorch models:** `.pt` files
- **ONNX models:** `.onnx` files
- **Batch processing:**
  - `.pt` models: 500-image batches
  - `.onnx` models: Single-image batches (fixed batch size limitation)

### Auto-Deletion Feature

- **Action:** Images with **zero detections** are automatically removed
- **Destination:** Moved to `./deleted/unlabeled/`
- **Purpose:** Reduce manual filtering effort
- **Configurable:** Can disable via `autolabel_delete_unlabeled: false`

### Label Generation

- **Format:** JSON (LabelMe/X-AnyLabeling compatible)
- **Structure:**
  - `shapes[]` array with bounding boxes
  - `label` field with class name
  - `points` field with 4-point rectangle: `[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]`
  - `description` field with confidence score: `"conf:0.87"`
  - `shape_type: "rectangle"`
- **Metadata:**
  - `imagePath` (filename)
  - `imageWidth`, `imageHeight`
  - `version: "0.4.0"`

### GPU Acceleration

- **Detection:** Automatic CUDA availability check
- **Warnings (red text):**
  - "WARNING: CUDA not available - running on CPU"
  - "This will be significantly slower than GPU inference"
- **Status display:**
  - GPU name and memory (e.g., "GPU detected: NVIDIA RTX 3090 (24.0 GB)")
  - Device confirmation: "✓ Using GPU for inference" or "✗ Using CPU for inference"
- **Device override:** Via config `autolabel_device` ("cpu", "cuda", "0", etc.)

### Inference Parameters

- **Confidence threshold:** 0.25 (configurable)
- **IoU threshold:** 0.45 (configurable, for NMS)
- **Image size:** 640 (configurable, input size for model)

### Statistics & Progress

- **Real-time progress bar** (tqdm)
- **Summary report:**
  - Total images processed
  - Images with detections (labeled)
  - Images without detections (unlabeled)
  - Images removed/deleted
  - Total detections across all images
  - **Per-class breakdown:** Detection counts for each class

### Configuration Keys

```json
{
  "yolo_model_path": "",
  "autolabel_input_dir": "./extracted_frames",
  "autolabel_confidence": 0.25,
  "autolabel_iou": 0.45,
  "autolabel_imgsz": 640,
  "autolabel_device": "",  // "" = auto-detect
  "autolabel_delete_unlabeled": true,
  "autolabel_deleted_dir": "./deleted/unlabeled"
}
```

---

## Step 3: Anonymize

**File:** `anonymize.py`
**Purpose:** Blur faces and license plates for privacy compliance (GDPR, dataset sharing).

### Detection Types

- **Face detection:** Threshold 0.3 (configurable)
- **License plate detection:** Threshold 0.3 (configurable)
- **Library:** `understand-ai/anonymizer`

### Obfuscation Method

- **Gaussian blur** with configurable parameters
- **Kernel format:** `"kernel_size,sigma,box_kernel_size"`
- **Default:** `"65,3,19"`
  - Kernel size: 65 (main blur)
  - Sigma: 3 (Gaussian standard deviation)
  - Box kernel size: 19 (pre-blur for efficiency)

### Library Integration

- **Automatic weights download:** To `./anonymizer_weights/`
- **Detector initialization:** Separate detectors for face and plate
- **Obfuscator configuration:** Custom kernel parameters
- **File type filtering:** Only processes configured image extensions

### Graceful Fallback

If library not installed:
- **Red warning:** "ANONYMIZER LIBRARY NOT FOUND"
- **Installation instructions:**
  - Git clone command
  - Pip install requirements
- **CLI alternative:** Full command with all parameters
- **Skip option:** User can skip anonymization and continue
  - Uses original images for next step
  - No anonymization applied

### Subdirectory Handling

- **Lists available folders** in base directory
- **Numeric selection:** Type `1`, `2`, `3` to select folder
- **Default suggestion:** Based on video name or single folder
- **Video name detection:** From previous steps or config `video_path`

### Configuration Keys

```json
{
  "anonymize_input_dir": "./kept_images",
  "anonymize_output_dir": "./anonymized_images",
  "anonymizer_weights_dir": "./anonymizer_weights",
  "face_threshold": 0.3,
  "plate_threshold": 0.3,
  "obfuscation_kernel": "65,3,19",
  "image_extensions": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
}
```

---

## Step 4: Review/Correct Labels

**File:** `labeling.py`
**Purpose:** Launch X-AnyLabeling GUI for manual annotation or correction of auto-labels.

### Virtual Environment Detection

**Searches in order:**
1. `~/x-anylabeling_env/`
2. `~/venvs/x-anylabeling_env/`
3. `~/.venvs/x-anylabeling_env/`
4. `./x-anylabeling_env/`
5. Custom path from config

**Validation:** Checks for `bin/activate` script

### Clean Environment Setup

**Problem:** Conflicting Qt/OpenCV libraries from other venvs
**Solution:** Builds clean environment with:

- **Kept variables:** HOME, USER, LANG, TERM, DISPLAY, WAYLAND_DISPLAY, XDG_RUNTIME_DIR, DBUS_SESSION_BUS_ADDRESS, SHELL, SSH_*
- **Cleaned PATH:** Strips all venv/env paths, keeps only system paths
- **Custom PYTHONPATH:** Points to X-AnyLabeling repo

### Auto-Launch

- **Activation:** Sources venv activate script
- **Command:** `python -m anylabeling.app`
- **Process monitoring:** Shows PID, waits for GUI to close
- **Working directory:** `~/` (user home)

### Manual Fallback

If venv not found:
- **Lists all searched paths**
- **Shows manual launch commands:**
  1. Activate venv
  2. Run anylabeling
  3. Open specific directory
- **Wait prompt:** Press Enter when done

### Empty Label Cleanup (Post-Labeling)

**Prompt:** "Remove images with empty or missing labels? (y/n)"

#### Detection Logic

- **JSON:** Empty `shapes[]` array
- **TXT (YOLO):** Blank file or whitespace-only
- **XML (Pascal VOC):** Zero `<object>` elements

#### Action

- **Moves (not deletes):**
  - Image file → `./deleted/empty_labels/`
  - Label file → `./deleted/empty_labels/`
- **Counts:** Shows number of moved images
- **No empty labels:** "All images have annotations"

#### User Interaction

- User specifies directory to clean (default: labeling input dir)
- Directory must exist (red warning if not)
- Uses config `label_format` to determine which extension to check

### Configuration Keys

```json
{
  "labeling_input_dir": "./anonymized_images",
  "anylabeling_venv": "x-anylabeling_env",
  "anylabeling_repo": "~/AI_Hub/X-AnyLabeling",
  "deleted_empty_labels_dir": "./deleted/empty_labels",
  "label_format": "json",
  "image_extensions": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
}
```

---

## Step 5: Consolidate + YOLO Conversion

**File:** `consolidate.py` + `convert_json_to_yolo.py`
**Purpose:** Re-consolidate corrected labels into separated folders and convert to YOLO training format.

### Folder Structure

- **Mode:** `separate_folders: true`
- **Output:**
  - `Dataset/Image/` - All images
  - `Dataset/Label/` - All labels
- **Intermediate (removed after YOLO conversion):**
  - Folders above are deleted after conversion
- **Final YOLO structure:**
  - `Dataset/images/train/`
  - `Dataset/images/val/`
  - `Dataset/labels/train/`
  - `Dataset/labels/val/`
  - `Dataset/classes.txt`
  - `Dataset/dataset.yaml`

### Multi-Format Support

#### JSON (LabelMe/X-AnyLabeling)
- Extension: `.json`
- Update: `imagePath` field
- Structure: `shapes[]` with `label`, `points`, `shape_type`

#### TXT (YOLO)
- Extension: `.txt`
- Format: `class_id x_center y_center width height` (normalized 0-1)
- No updates needed (copied as-is)

#### XML (Pascal VOC)
- Extension: `.xml`
- Update: `<filename>` and `<path>` tags
- Structure: `<object>` with `<name>` and `<bndbox>`

### Format Detection

**Automatic scanning:**
- Walks directory tree
- Counts files per format
- **Validates YOLO TXT:** Checks first line has 5 numbers (class x y w h)
- **Suggests most common format**

**User selection menu:**
```
Select label format to process:
  1. JSON (LabelMe/X-AnyLabeling) - 150 files
  2. TXT (YOLO) - 0 files
  3. XML (Pascal VOC) - 0 files

Suggested: Option 1 (JSON (LabelMe/X-AnyLabeling))
Enter choice (1-3) [default: 1]:
```

### Overwrite Protection

**Warning display (red text):**
```
WARNING: Output directory './Dataset' already exists.
  Existing files: 302 (151 images)
  New files will be added with sequential numbering after existing ones.
```

**Behavior:**
- Scans existing files matching pattern `Image\d+.*`
- Finds max index (e.g., `Image00000150.jpg` → index 150)
- Starts new numbering at `max_index + 1`
- **Does not overwrite** existing files

### Interactive Configuration

**Preview display:**
```
------------------------------------------------------------
  1. Input:  ./anonymized_images/video_name
  2. Output: ./Dataset
     Labels: JSON (LabelMe/X-AnyLabeling) -> YOLO TXT (auto-converted)
     Mode:   Copy
------------------------------------------------------------
[1=change input, 2=change output, 3=add input, Enter=proceed, q=cancel]:
```

**Options:**
- **1:** Change input directory
  - If multiple inputs: Choose which to replace or remove (`r1`, `r2`)
  - Enter new path or press Enter to keep
- **2:** Change output directory
- **3:** Add additional input directory (multiple sources)
- **Enter:** Proceed with consolidation
- **q:** Cancel operation

### Label Processing

#### JSON Updates
```json
{
  "imagePath": "Image00000042.jpg",  // Updated to new filename
  "shapes": [...],
  "imageWidth": 1920,
  "imageHeight": 1080
}
```

#### XML Updates
```xml
<filename>Image00000042.jpg</filename>  <!-- Updated -->
<path>Image00000042.jpg</path>          <!-- Updated -->
```

#### YOLO TXT
- No modifications
- Direct copy to Label folder

### Error Handling

**Label validation:**
- Labels processed **before** moving files
- JSON parse errors caught
- XML parse errors caught
- **Red warning** on failure: "Warning: Failed to process label"
- **Skips entire image-label pair** on error
- Maintains dataset integrity (no orphaned images)

**Failed file tracking:**
- Counts failed operations
- Shows in summary

### File Operations

**Copy mode (default):**
- `shutil.copy2()` preserves timestamps
- Original files untouched
- Safe for iterative workflows

**Move mode:**
- `shutil.move()` relocates files
- Original directory emptied
- Faster, saves disk space

### Progress Display

**In-place updates:**
```
[42/150] old_image_name.jpg -> Image00000042.jpg (+label)
```

- Updates every 10 files (or all if ≤20 total)
- Shows label status: `(+label)` or no indicator
- Carriage return for in-place update

**Summary:**
```
============================================================
CONSOLIDATION SUMMARY
============================================================
Total images found: 150
Successfully processed: 148
Failed: 2
Label files processed: 148
Images without labels: 0
Label format: JSON (LabelMe/X-AnyLabeling)

Consolidated images saved to: ./Dataset/Image
Label files saved to: ./Dataset/Label
============================================================
```

### YOLO Conversion (Automatic)

**Triggered by:** `config["convert_to_yolo"] = true` (set by workflow)

**Process:**
1. Reads consolidated `Image/` and `Label/` folders
2. Extracts unique classes from all JSON labels
3. Creates train/val split (default: 80/20)
4. Converts JSON boxes to YOLO normalized format
5. Generates `classes.txt` (one class per line)
6. Generates `dataset.yaml`:
   ```yaml
   path: /absolute/path/to/Dataset
   train: images/train
   val: images/val
   nc: 3
   names: ['class1', 'class2', 'class3']
   ```
7. **Removes intermediate folders:** `Image/` and `Label/`
8. Outputs training command: `python train_yolo.py --data /path/to/dataset.yaml`

**Split method:**
- Randomized with fixed seed (reproducible)
- Respects `yolo_train_ratio` (0.8 = 80% train, 20% val)

**Error handling:**
- Red warning on conversion failure
- Returns false to halt workflow
- Preserves intermediate folders on error

### Configuration Keys

```json
{
  "consolidated_output_dir": "./Dataset",
  "include_labels": true,
  "copy_files": true,
  "separate_folders": true,
  "label_format": "json",
  "skip_format_prompt": true,  // Set by workflow
  "convert_to_yolo": true,     // Set by workflow
  "yolo_train_ratio": 0.8,
  "yolo_classes_file": null,   // Optional pre-defined classes
  "image_extensions": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
}
```

---

## Step 6: Train YOLO Model

**File:** `train_yolo.py` (imported from YOLO_Training/)
**Purpose:** Train YOLOv8 object detection model using the converted YOLO dataset.

### Prerequisites

- **Step 5 completion:** Requires `dataset.yaml` from YOLO conversion
- **YOLO dataset structure:**
  - `images/train/` and `images/val/`
  - `labels/train/` and `labels/val/`
  - `dataset.yaml` with class definitions
  - `classes.txt` with class names

### Model Selection

**Pre-trained models (YOLOv8):**
- **yolov8n.pt** (Nano): Fastest, smallest, lowest accuracy
- **yolov8s.pt** (Small): Balanced speed and accuracy
- **yolov8m.pt** (Medium): Better accuracy, slower
- **yolov8l.pt** (Large): High accuracy, resource-intensive
- **yolov8x.pt** (Extra-large): Best accuracy, slowest

**Custom models:**
- Path to previously trained `.pt` file
- Resume training from checkpoint

### Training Parameters

#### Core Settings
- **Epochs:** Number of training iterations (default: 100)
  - More epochs = better learning (if not overfitting)
  - Watch validation metrics to determine optimal value
- **Batch size:** Images per training step (default: 16)
  - Higher = faster training, more GPU memory required
  - Lower = slower, more stable gradients
  - `-1` = auto-batch (automatically determines max batch size)
- **Image size:** Input dimensions for training (default: 640)
  - Common values: 320, 416, 512, 640, 1280
  - Higher = better detection of small objects, slower training

#### Device Configuration
- **Device:** CUDA device selection
  - Empty string = auto-detect (uses GPU if available)
  - `"0"` = First GPU
  - `"0,1,2,3"` = Multi-GPU training
  - `"cpu"` = Force CPU (slow, for testing)
- **Workers:** Dataloader worker threads (default: 8)
  - More workers = faster data loading
  - Optimal: 4-8 per GPU

#### Optimization
- **Optimizer:** Training optimizer (default: auto)
  - `"SGD"`: Classic stochastic gradient descent
  - `"Adam"`: Adaptive learning rate
  - `"AdamW"`: Adam with weight decay
  - `"auto"`: Automatically selects based on model
- **Learning rate (lr0):** Initial learning rate (default: 0.01)
  - Higher = faster convergence, risk of instability
  - Lower = stable training, slower convergence
- **Patience:** Early stopping patience (default: 50)
  - Stops training if no improvement for N epochs
  - `0` = disable early stopping

#### Performance
- **AMP (Automatic Mixed Precision):** Default enabled
  - Uses FP16 for faster training and lower memory usage
  - Minimal accuracy impact
  - Can disable with `train_amp: false`
- **Cache:** Cache images in RAM (default: disabled)
  - Speeds up training significantly
  - Requires sufficient RAM for entire dataset
  - Enable with `train_cache: true`

#### Augmentation
- **Data augmentation:** Default enabled
  - Randomly modifies training images to improve generalization
  - Includes: HSV shifts, flips, translation, scaling, mosaic
  - Configured parameters:
    - `hsv_h: 0.015` (Hue shift)
    - `hsv_s: 0.7` (Saturation shift)
    - `hsv_v: 0.4` (Value/brightness shift)
    - `degrees: 0.0` (Rotation - disabled by default)
    - `translate: 0.1` (Translation)
    - `scale: 0.5` (Scaling)
    - `fliplr: 0.5` (Horizontal flip probability)
    - `mosaic: 1.0` (Mosaic augmentation - combines 4 images)

### Training Process

**Display before training:**
```
==================================================
YOLOv8 Training Configuration
==================================================
  Model:      yolov8n.pt
  Dataset:    ./Dataset/dataset.yaml
  Epochs:     100
  Batch size: 16
  Image size: 640
  Device:     auto
  Project:    ./runs
  Name:       yolov8n_20260204_233000
==================================================
```

**Live progress:**
- Epoch counter
- Training loss (box, cls, dfl)
- Validation metrics (mAP50, mAP50-95)
- Time per epoch
- ETA

**Automatic plots generated:**
- Confusion matrix
- PR curves (Precision-Recall)
- F1 curves
- Training/validation loss curves
- mAP curves

### Post-Training Features

#### Sample Visualization (Interactive Prompt)

**Prompt:** "Run sample visualization on validation images? [Y/n]:"

**If yes:**
- Asks for number of samples (default: 20)
- Asks for confidence threshold (default: 0.33)
- Randomly selects N validation images
- Runs inference with best model
- Draws bounding boxes with class labels and confidence
- Saves annotated images to `<results_dir>/sample_predictions/`

**Output example:**
```
Visualizing 20 sample predictions (conf=0.33)...
  [1/20] image_001.jpg: 3 detections - person(0.92), car(0.87), bike(0.45)
  [2/20] image_002.jpg: 1 detection - dog(0.78)
  ...
  [20/20] image_020.jpg: 0 detections - none

Sample predictions saved to: ./runs/yolov8n_20260204/sample_predictions
```

#### ONNX Export (Interactive Prompt)

**Prompt:** "Export model to ONNX format? [Y/n]:"

**If yes:**
- Exports `best.pt` to `best.onnx`
- ONNX format benefits:
  - Cross-platform compatibility
  - Faster inference on some hardware
  - No PyTorch dependency for deployment
  - Required for X-AnyLabeling integration

**Auto-generates X-AnyLabeling config:**
- File: `<results_dir>/weights/x_anylabeling_config.yaml`
- Contains:
  - Model path (ONNX)
  - Input dimensions
  - Thresholds (confidence, IoU, NMS)
  - Class names list
- **Ready to use:** Copy to X-AnyLabeling models directory

**Example config:**
```yaml
type: yolov8
name: yolov8n_20260204
display_name: yolov8n_20260204
model_path: /path/to/best.onnx
input_width: 640
input_height: 640
score_threshold: 0.25
confidence_threshold: 0.25
nms_threshold: 0.45
iou_threshold: 0.45
classes:
  - person
  - car
  - dog
```

### Output Structure

**Results directory:** `./runs/<name>/`

```
runs/
└── yolov8n_20260204_233000/
    ├── weights/
    │   ├── best.pt              # Best model (by mAP)
    │   ├── last.pt              # Latest epoch
    │   ├── best.onnx            # ONNX export (if exported)
    │   └── x_anylabeling_config.yaml  # Config (if exported)
    ├── sample_predictions/      # Validation samples (if visualized)
    │   ├── sample_1_image_001.jpg
    │   ├── sample_2_image_002.jpg
    │   └── ...
    ├── confusion_matrix.png
    ├── F1_curve.png
    ├── PR_curve.png
    ├── P_curve.png
    ├── R_curve.png
    ├── results.csv              # Metrics per epoch
    ├── results.png              # Training curves
    └── args.yaml                # Training arguments used
```

### Resume Training

**Enable with:** `train_resume: true`

**Behavior:**
- Looks for `<project>/<name>/weights/last.pt`
- Resumes from last checkpoint if found
- Starts fresh if checkpoint doesn't exist
- Preserves:
  - Optimizer state
  - Learning rate schedule
  - Epoch counter
  - Best mAP record

### Error Handling

**Common issues:**

1. **Dataset YAML not found:**
   - Red error: "dataset.yaml not found. Run step 5 first."
   - Solution: Complete Step 5 (Consolidate & convert to YOLO)

2. **CUDA out of memory:**
   - Error: "CUDA out of memory"
   - Solution: Reduce batch size or image size
   - Try: `train_batch: 8` or `train_imgsz: 416`

3. **No training images:**
   - Error: "Dataset has no training images"
   - Solution: Check `dataset.yaml` paths, ensure images exist

4. **Ultralytics not installed:**
   - Red error: "YOLO training module not available"
   - Solution: `pip install ultralytics`

### Integration with Workflow

**Step chaining:**
- Automatically uses `dataset.yaml` from Step 5
- Stores results directory in `config["training_results_dir"]`
- Shows path in workflow summary

**Success criteria:**
- Training completes without crashes
- Weights saved to `runs/<name>/weights/`
- Returns true to workflow

**Failure behavior:**
- Shows error traceback
- Returns false (stops workflow or prompts to continue)
- Preserves partial training results

### Configuration Keys

```json
{
  "train_model": "yolov8n.pt",
  "train_epochs": 100,
  "train_batch": 16,
  "train_imgsz": 640,
  "train_device": "",
  "train_workers": 8,
  "train_project": "./runs",
  "train_name": null,
  "train_resume": false,
  "train_pretrained": true,
  "train_optimizer": "auto",
  "train_lr0": 0.01,
  "train_patience": 50,
  "train_cache": false,
  "train_amp": true,
  "train_augment": true
}
```

### Tips

**Model selection:**
- Start with `yolov8n.pt` for quick experiments
- Use `yolov8s.pt` for production (good balance)
- Use larger models only if accuracy is critical

**Batch size:**
- Try largest batch that fits in GPU memory
- Use `-1` for auto-batch detection
- Reduce if out-of-memory errors occur

**Epochs:**
- 100 epochs is a good starting point
- Watch validation mAP - stop when it plateaus
- Use early stopping (patience) to avoid overfitting

**Learning rate:**
- Default (0.01) works well for most cases
- Lower (0.001) if training is unstable
- Higher (0.1) for small datasets or fine-tuning

**Augmentation:**
- Keep enabled for small datasets (prevents overfitting)
- Can disable for very large datasets to speed up training

**Validation:**
- Always visualize sample predictions to check quality
- Review confusion matrix for class-specific issues
- Check PR curves to understand precision/recall tradeoff

---

## Alternative Step 6: Review Labels Viewer

**File:** `review_labels.py`
**Purpose:** Visually inspect labeled images with annotation overlays.

**Note:** This is an alternative to final consolidation. Used in standard pipeline, not YOLO training workflow.

### Multi-Format Support

- **JSON (LabelMe):** Reads `shapes[]` directly
- **TXT (YOLO):** Converts normalized coords to pixels
- **XML (Pascal VOC):** Parses `<object>` and `<bndbox>` elements

**Format detection:** Same as consolidate step (auto-scan + user selection)

### Folder Structure Detection

**Checks for separated structure:**
```
Dataset/
├── Image/
└── Label/
```

**Falls back to flat structure:**
```
Dataset/
├── image1.jpg
├── image1.json
├── image2.jpg
└── image2.json
```

### Interactive Viewer

**Window:** OpenCV resizable window (max: 1400×900)

**Visual elements:**
- **Bounding boxes:** 2-pixel thick rectangles
- **Class labels:** White text on colored background above each box
- **Legend:** Top-left corner showing all classes with colors
- **Info bar:** Bottom overlay (semi-transparent black)
  - Format: `[idx/total] filename | Label: label_filename.json`
  - Updates per image

**Color coding:**
- 10 distinct colors: Green, Red, Blue, Cyan, Magenta, Yellow, Orange, Purple, Dark Green, Olive
- Consistent per class (same class = same color)
- Auto-assigned on first appearance

### Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE`, `D`, `→` | Next image |
| `A`, `←` | Previous image |
| `S` | Save current view with overlays |
| `Q`, `ESC` | Quit viewer |

**Window close:** X button also quits

### Error Resilience

**Failed image loads:**
- Caught via `cv2.imread() == None`
- Automatically skips to next image
- Tracked in `failed_images` set (prevents duplicate warnings)
- **Infinite loop prevention:** Exits if all images fail

**Failed label loads:**
- JSON parse errors
- XML parse errors
- Missing label files
- Shows error in info bar: `"Error loading label: <message>"`
- Image still displayed (without annotations)

**Summary at end:**
```
Warning: 3 image(s) failed to load:
  - /path/to/corrupt_image1.jpg
  - /path/to/corrupt_image2.jpg
  - /path/to/corrupt_image3.jpg
```
- Shows first 10, then "... and X more"

### Save Feature

**Trigger:** Press `S` key

**Output:**
- Filename: `review_<original_name>`
- Location: `config["review_output_dir"]`
- Content: Current view with all overlays (boxes, labels, legend, info bar)
- Format: Same as original (JPG/PNG)

**Feedback:** Prints `"Saved: <path>"` to console

### Configuration Keys

```json
{
  "review_input_dir": "./Dataset",
  "review_output_dir": "./review_screenshots",
  "label_format": "json"  // or "txt", "xml"
}
```

---

## General Features

### Features Present in All Steps

#### 1. Subdirectory Listing & Selection

**Display:**
```
Available folders in ./extracted_frames:
  1. video_name_1
  2. video_name_2
  3. video_name_3

Enter directory [default: ./extracted_frames/video_name_1]:
```

**Input options:**
- **Numeric:** Type `1`, `2`, `3` to select folder
- **Custom path:** Type full path
- **Default:** Press Enter to use suggested default

**Default logic:**
- If `video_name` in config and exists → use it
- If only one subdirectory → use it
- Otherwise → use base directory

#### 2. Default Value Prompts

**Format:** `[default: value]`

**Examples:**
```
Enter confidence threshold [default: 0.25]:
Enter output directory [default: ./Dataset]:
```

**Behavior:** Press Enter to accept default

#### 3. Red Warning Messages

**Requirement:** Per project guidelines in CLAUDE.md

**ANSI code:** `\033[91m` (RED) with `\033[0m` (RESET)

**Examples:**
- `"ERROR: File not found"`
- `"WARNING: Output directory already exists"`
- `"CUDA not available - running on CPU"`
- `"Failed to process label file"`

#### 4. From Previous Step Integration

**Parameter:** `from_previous_step: bool`

**Logic:**
```python
if from_previous_step and config.get("previous_output_key"):
    input_dir = config["previous_output_key"]
else:
    # Prompt user for input
```

**Benefits:**
- Seamless step chaining
- Reduces user prompts
- Maintains workflow state

#### 5. Video Name Tracking

**Sources:**
1. `config["video_name"]` (set by previous steps)
2. `Path(config["video_path"]).stem` (extracted from video path)

**Usage:**
- Default subdirectory selection
- Output folder organization
- Consolidation naming

#### 6. Deleted Files (Not Permanently Deleted)

**Locations:**
- `./deleted/clustering/` (blurry/similar frames)
- `./deleted/unlabeled/` (no YOLO detections)
- `./deleted/empty_labels/` (no annotations after labeling)

**Benefit:** Recoverable if needed, audit trail

#### 7. Continue/Cancel Prompts on Failure

**Pattern:**
```python
if not success:
    continue_choice = input("Continue to next step anyway? (y/n): ")
    if continue_choice != 'y':
        print("Workflow stopped.")
        return False
```

**Benefit:** User control over workflow progression

---

## Workflow Progress Display

### Visual Progress Indicators

**Symbols:**
- **✓** (Green checkmark): Completed steps
- **▶** (Yellow arrow): Current step (bold text)
- **○** (Dim circle): Pending steps
- **─** (Dim dash): Skipped steps (started mid-workflow)

**Example:**
```
==================================================
  ─ Step 1: Extracting image frames from video
  ✓ Step 2: YOLO auto-labeling
  ▶ Step 3: Anonymizing faces and license plates
  ○ Step 4: Consolidating files
  ○ Step 5: Reviewing/correcting labels (X-AnyLabeling)
  ○ Step 6: Consolidating files (final)
==================================================
```

### Summary Report

**After workflow completion:**
```
============================================================
  WORKFLOW SUMMARY
============================================================
  ✓ Step 1: Extracting image frames from video
      → ./extracted_frames/video_name
  ✓ Step 2: YOLO auto-labeling
      → ./extracted_frames/video_name
  ✓ Step 3: Anonymizing faces and license plates
      → ./anonymized_images/video_name
  ✓ Step 4: Reviewing/correcting labels (X-AnyLabeling)
      → ./anonymized_images/video_name
  ✓ Step 5: Consolidating & converting to YOLO format
      → ./Dataset
  ✓ Step 6: Training YOLO model
      → ./runs/yolov8n_20260204_233000

  YOLO dataset: /path/to/Dataset/dataset.yaml
============================================================
```

**Contents:**
- Step number and name
- Success/failure icon
- Output directory for each step
- YOLO dataset path (if conversion performed)

---

## Configuration File Structure

### `yolo_workflow_config.json` (Complete Example)

```json
{
  "video_path": "",
  "extracted_frames_dir": "./extracted_frames",
  "frame_threshold": 0.85,
  "target_fps": 3.0,
  "frame_interval": null,
  "max_frames": null,
  "histogram_bins": 32,
  "frame_prefix": "frame",
  "blur_threshold": 100.0,
  "clustering": false,
  "clustering_eps": null,
  "clustering_min_samples": 2,

  "yolo_model_path": "",
  "autolabel_input_dir": "./extracted_frames",
  "autolabel_confidence": 0.25,
  "autolabel_iou": 0.45,
  "autolabel_imgsz": 640,
  "autolabel_device": "",
  "autolabel_delete_unlabeled": true,
  "autolabel_deleted_dir": "./deleted/unlabeled",

  "anonymize_input_dir": "./extracted_frames",
  "anonymize_output_dir": "./anonymized_images",
  "anonymizer_weights_dir": "./anonymizer_weights",
  "face_threshold": 0.3,
  "plate_threshold": 0.3,
  "obfuscation_kernel": "65,3,19",

  "consolidated_output_dir": "./Dataset",
  "include_labels": true,
  "copy_files": true,
  "label_format": "json",
  "image_extensions": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],

  "labeling_input_dir": "./anonymized_images",
  "anylabeling_venv": "x-anylabeling_env",
  "anylabeling_repo": "~/AI_Hub/X-AnyLabeling",

  "yolo_train_ratio": 0.8,
  "yolo_classes_file": null,

  "train_model": "yolov8n.pt",
  "train_epochs": 100,
  "train_batch": 16,
  "train_imgsz": 640,
  "train_device": "",
  "train_workers": 8,
  "train_project": "./runs",
  "train_name": null,
  "train_resume": false,
  "train_pretrained": true,
  "train_optimizer": "auto",
  "train_lr0": 0.01,
  "train_patience": 50,
  "train_cache": false,
  "train_amp": true,
  "train_augment": true
}
```

---

## Command-Line Usage

### Basic Usage

```bash
# Interactive mode (prompts for all inputs)
python yolo_training_workflow.py

# With video and model specified
python yolo_training_workflow.py --video video.mp4 --model best.pt

# Start from specific step (skip earlier steps)
python yolo_training_workflow.py --start-step 3

# Use custom config file
python yolo_training_workflow.py --config my_config.json

# Generate template config
python yolo_training_workflow.py --generate-config
```

### Arguments

- `--video` / `-v`: Path to input video file (for step 1)
- `--model` / `-m`: Path to YOLO model for auto-labeling (.pt or .onnx, for step 2)
- `--start-step` / `-s`: Start from step 1-6
  - `1`: Extract frames
  - `2`: YOLO auto-label
  - `3`: Anonymize
  - `4`: Review/correct labels
  - `5`: Consolidate & convert to YOLO
  - `6`: Train YOLO model
- `--config` / `-c`: Path to JSON config file
- `--generate-config` / `-g`: Create template config and exit

### Config File Discovery

**Search order:**
1. `--config` argument path
2. `./yolo_workflow_config.json`
3. `<script_dir>/yolo_workflow_config.json`
4. `./workflow_config.json`
5. `<script_dir>/workflow_config.json`

**Fallback:** Built-in defaults if no config found

---

## Tips & Best Practices

### Frame Extraction
- Use **sequential clustering** for video (faster, sufficient)
- Use **DBSCAN** for mixed image sources (more thorough)
- Increase `blur_threshold` if too many blurry frames pass
- Decrease `frame_threshold` for more strict deduplication (fewer frames)

### YOLO Auto-Labeling
- Use `.pt` models for faster batch inference
- Lower `autolabel_confidence` to catch more objects (more false positives)
- Check GPU usage - CPU inference is 10-100× slower
- Review per-class counts to identify underperforming classes

### Anonymization
- Skip if dataset is for internal use only
- Test thresholds on sample images first
- Lower thresholds (0.2) for more aggressive blurring
- Higher thresholds (0.5) to reduce false positives

### Consolidation
- Use **copy mode** for safety (can re-run if needed)
- Use **move mode** to save disk space (cannot undo)
- Add multiple input directories to merge datasets
- Check overwrite warnings carefully

### Review/Correction
- Focus on false positives (incorrect boxes) and false negatives (missed objects)
- Use X-AnyLabeling shortcuts to speed up corrections
- Remove empty-label images after review (reduces dataset size)

### YOLO Training (Step 6)
- Start with small model (yolov8n) for quick experiments
- Aim for 80/20 or 70/30 train/val split
- Ensure all classes have sufficient samples (50+ per class minimum)
- Monitor validation mAP - stop if it plateaus or decreases
- Use early stopping (patience) to prevent overfitting
- Always run sample visualization to verify model quality
- Export to ONNX for deployment and X-AnyLabeling integration

---

## Troubleshooting

### "No video or image files found"
- Check video file extensions (supported: mp4, avi, mov, mkv, etc.)
- Check image file extensions (supported: jpg, png, bmp, tiff, webp)
- Verify path is correct and accessible

### "CUDA not available"
- Install PyTorch with CUDA support
- Check NVIDIA driver installation
- Verify GPU is detected: `nvidia-smi`
- Override with `autolabel_device: "cuda"`

### "X-anylabeling venv not found"
- Check venv name in config matches actual venv
- Verify venv has X-AnyLabeling installed
- Use manual fallback instructions

### "Failed to process label"
- Label file may be corrupt (invalid JSON/XML)
- Check label format matches config setting
- Manually inspect problematic label file

### "All images failed to load" (Review step)
- Check image file corruption
- Verify image paths are correct
- Check file permissions

---

## Output Directory Structure

```
Project/
├── extracted_frames/
│   └── video_name/
│       ├── frame_000000.png
│       ├── frame_000001.png
│       └── ...
├── anonymized_images/
│   └── video_name/
│       ├── frame_000000.png
│       ├── frame_000000.json
│       └── ...
├── Dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── classes.txt
│   └── dataset.yaml
└── deleted/
    ├── clustering/
    ├── unlabeled/
    └── empty_labels/
```

---

## Version History

- **v6.0** - Added Step 6 (YOLO Training) to workflow
- **v5.0** - Added YOLO training workflow
- **v4.0** - Sequential clustering, parallel processing
- **v3.0** - Multi-format support, overwrite protection
- **v2.0** - Anonymization, X-AnyLabeling integration
- **v1.0** - Initial frame extraction and consolidation

---

**Document created:** 2026-02-04
**Last updated:** 2026-02-04 23:30 KST
