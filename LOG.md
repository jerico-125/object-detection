# Change Log

---

## 2026-02-12

- **Streamlined model export in training step**:
  - Modified: `Pipeline/train.py` — Combined the separate ONNX and TensorRT export prompts into a single "Export model?" prompt. When a CUDA GPU is available, runs `export(format='engine')` which automatically produces both ONNX and TensorRT engine in one step. When no GPU is detected, falls back to ONNX-only export with explicit `format='onnx'` and `imgsz`. Removed the redundant standalone ONNX export that was running before the TensorRT export.

- **Improved inference output naming and added image/folder support**:
  - Modified: `Pipeline/inference.py` — Video outputs now saved as `<original_name>_labeled.mp4` next to the source (instead of `output_<timestamp>.mp4`). Added image inference support: single images and image folders are now valid `--source` inputs. Folder sources save annotated images to a sibling `<folder>_labeled/` directory. Single image sources also save to a `<parent_folder>_labeled/` directory. Added `_video_output_path()`, `_is_image()`, `_is_video()`, and `run_image_inference()` functions. Updated module docstring, argparse help/epilog, and CLI examples. Removed unused `os` import.
  - Modified: `Pipeline/main.py` — Updated Step 7 title from "Video inference" to "Inference (video / image / folder)" across menu, step list, epilog, docstring, and progress display. Updated input prompt to mention image/folder sources.

- **Auto-enable `--show` for webcam sources**:
  - Modified: `Pipeline/inference.py` — When `--source` is a camera index (integer) and `--show` is not passed, `--show` is now auto-enabled with a yellow info message. Prevents useless batch-mode processing of a live webcam feed.

- **Added TensorRT export after training**:
  - Modified: `Pipeline/train.py` — After ONNX export, prompts user to also export to TensorRT `.engine` format (FP16). Only offered when a CUDA GPU is detected. Added `import torch` for GPU detection.

- **Model selector prefers TensorRT engine by default**:
  - Modified: `Pipeline/model_utils.py` — `find_yolo_versions()` now checks for `best.engine` before `best.pt`, so TensorRT models are used by default when available. Version listing now shows the format tag (`[engine]` / `[pt]`) next to each version. Custom path prompt updated to mention `.engine`.

**Git commit 586e591** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

## 2026-02-12

- **Integrated video inference into main.py as Step 7**:
  - Modified: `Pipeline/main.py` — imported `run_inference` from `inference.py`, added inference config defaults (`inference_conf`, `inference_iou`, `inference_imgsz`, `inference_show`, `inference_save`, `inference_device`, `inference_half`), added `run_inference_step()` wrapper that prompts for video source and model selection, updated all step numbering from 6 to 7 steps (menu, input validation, argparse choices, workflow loop, progress display, docstrings/epilog), wired `trained_model_path` from Step 6 training results so Step 7 defaults to the freshly trained model
  - Modified: `Pipeline/inference.py` — added `return True` at end of `run_inference()` so workflow can track success/failure; fixed `total_mem` → `total_memory` in `print_device_info()` for PyTorch 2.10+ compatibility

**Git commit 2250c43** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

## 2026-02-12

- **Deleted unused `Pipeline/__init__.py`**:
  - Deleted: `Pipeline/__init__.py`
  - Nothing in the codebase imports from the `Pipeline` package — all scripts use direct imports (`from extract import ...`) since they run from within the Pipeline directory
  - The file was also entirely broken (all imports referenced old `stepX_` prefixed filenames that no longer exist)

- **Renamed 5 Pipeline files for brevity**:
  - Renamed: `extract_frames.py` → `extract.py`, `train_yolo.py` → `train.py`, `video_inference.py` → `inference.py`, `yolo_training_workflow.py` → `main.py`, `workflow_config.json` → `config.json`
  - Modified: `Pipeline/main.py` — updated imports (`from extract import`, `from train import`), docstring usage examples, config file search paths, error messages
  - Modified: `Pipeline/train.py` — updated docstring usage examples
  - Modified: `Pipeline/inference.py` — updated docstring usage examples
  - Modified: `Pipeline/model_utils.py` — updated docstring file references
  - Modified: `Pipeline/consolidate.py` — updated training command hint
  - Modified: `Pipeline/config.json` — updated documentation section headers
  - Modified: `CLAUDE.md` — updated project structure tree, CLI examples, config reference
  - Modified: `WORKFLOW_FEATURES.md` — updated all file references, CLI examples, config file discovery paths

- **Slimmed down CLAUDE.md to eliminate overlap with WORKFLOW_FEATURES.md and workflow_config.json**:
  - Modified: `CLAUDE.md`
  - Removed the entire "Configuration Files" section (all config key listings for frame extraction, anonymization, labeling, YOLO auto-labeling, and YOLO training) — these are already self-documented in `workflow_config.json`'s `_documentation` section
  - Removed the "Annotation Formats" section (covered in WORKFLOW_FEATURES.md)
  - Consolidated the "Running the Pipeline" section — combined standard and YOLO workflow examples, removed duplicated step list
  - Consolidated "Training (Standalone)" into a compact section
  - Added reference link to WORKFLOW_FEATURES.md at the top
  - Updated project structure tree to include `yolo_training_workflow.py`, `video_inference.py`, `model_utils.py`, and `WORKFLOW_FEATURES.md`
  - Reduced file from 214 lines to 114 lines

- **Updated WORKFLOW_FEATURES.md to reflect all changes since Feb 4**:
  - Modified: `WORKFLOW_FEATURES.md`
  - Step 2: Added model selection via `model_utils.select_yolo_model()`, output path mirroring behavior, `yolo_runs_dir` and `autolabel_output_dir` config keys
  - Step 3: Documented new return values (`"skipped"`, `"stop"`) for workflow branching
  - Step 6: Replaced `train_model` config key with interactive model selection, added auto version naming (`YOLO_v<N>`), added venv auto-activation docs
  - New sections: Video Inference tool (`video_inference.py`), Dataset Registry (`dataset_registry.json`), Shared Utilities (`model_utils.py`)
  - General Features: Replaced "Red Warning Messages" with full color scheme table (Green/Red/Yellow/Cyan) and divider standards
  - Updated full config example: removed `yolo_model_path`/`train_model`, added `yolo_runs_dir`/`autolabel_output_dir`
  - Updated version history to v7.0 and last-updated date

- **Added dataset registry and documented it in CLAUDE.md**:
  - New file: `dataset_registry.json`
  - Modified: `CLAUDE.md`
  - JSON file at the project root tracking which source sets compose each YOLO dataset version
  - Added `dataset_registry.json` to the project structure tree and a "Dataset Registry" section in CLAUDE.md

---
**Git commit d4a40b3** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

- **Standardized terminal UI: consistent dividers and color-coded output**:
  - Modified: `Pipeline/train_yolo.py`, `Pipeline/yolo_training_workflow.py`, `Pipeline/extract_frames.py`, `Pipeline/autolabel.py`, `Pipeline/consolidate.py`, `Pipeline/anonymize.py`, `CLAUDE.md`
  - Added YELLOW and CYAN color constants to all pipeline files
  - Yellow for configuration info (parameters, paths, settings the user should verify)
  - Cyan for processing/status info (system actions, summaries, results)
  - Green for user input prompts (added missing GREEN to `train_yolo.py` prompts)
  - Standardized all dividers: `=` * 60 for major sections, `-` * 60 for sub-sections
  - Removed `#` * 70 and `=` * 70 dividers from `yolo_training_workflow.py`
  - Removed `-` * 50 dividers from `extract_frames.py` and `autolabel.py`
  - Removed duplicate local CYAN/YELLOW definitions in `yolo_training_workflow.py` (now module-level)
  - Updated CLAUDE.md UI Guidelines with full color scheme and divider rules

- **Added UI guideline for green user prompts** in `CLAUDE.md`:
  - Modified: `CLAUDE.md`
  - Added rule that all `input()` prompts must use green text (`\033[92m`)

- **Removed duplicate training configuration display** from `yolo_training_workflow.py`:
  - Modified: `Pipeline/yolo_training_workflow.py`
  - `train_yolo_model()` was printing training config (model, name, epochs, etc.) before calling `train_yolo()`, which prints the same information — removed the duplicate from the workflow

- **Renamed project root from `AI_Hub` to `Object_Detection`** in all hardcoded paths:
  - Modified: `Pipeline/model_utils.py`, `Pipeline/video_inference.py`, `Pipeline/train_yolo.py`, `Pipeline/autolabel.py`, `Pipeline/labeling.py`, `Pipeline/yolo_training_workflow.py`, `Pipeline/workflow_config.json`, `CLAUDE.md`, `WORKFLOW_FEATURES.md`
  - Updated all `DEFAULT_RUNS_DIR`, `DEFAULT_VENV_PATH`, `anylabeling_repo`, `yolo_runs_dir` defaults and fallbacks
  - Renamed `ai_hub_root` variable to `project_root` in `autolabel.py`
  - Updated comments and help text referencing `AI_Hub`

- **Created dataset registry file**:
  - New file: `dataset_registry.json`
  - JSON file at the project root that tracks the composition of each YOLO dataset version (which source sets were combined)
  - Initial entries: `Dataset_YOLO_v1` (Set1 + Set2), `Dataset_YOLO_v2` (Set3/Bbox_1)

## 2026-02-06

- **Fixed duplicate prompt when skipping anonymization step**:
  - Modified files: `Pipeline/anonymize.py`, `Pipeline/yolo_training_workflow.py`
  - When user typed 'y' to skip anonymization, they were prompted again with "Step 3 done. Continue to Step 4?"
  - Changed `anonymize.py` to return `"skipped"` instead of `True` when skipping, and `"stop"` instead of `False` when declining
  - Updated `run_workflow()` in `yolo_training_workflow.py` to handle `"skipped"` (move to next step without prompting) and `"stop"` (end workflow immediately)
  - Removed the "Please install anonymizer and try again." message when user declines to skip
  - Rationale: Eliminates redundant prompt after skipping a step

- **Autolabel output mirrors full input path relative to AI_Hub root**:
  - Modified files: `Pipeline/autolabel.py`, `Pipeline/workflow_config.json`, `CLAUDE.md`
  - Output path now preserves full directory structure relative to `/home/aidall/AI_Hub`
  - Example: input `/home/aidall/AI_Hub/Set3/BoundingBox/Bbox_1_new` → output `./autolabeled/Set3/BoundingBox/Bbox_1_new`
  - Falls back to just the folder name if input is not under AI_Hub
  - Updated documentation in `workflow_config.json` and `CLAUDE.md`
  - Rationale: Maintains full path context for organizing labeled datasets

- **Changed autolabel to save labels only in output directory (not in-place)**:
  - Modified file: `Pipeline/autolabel.py`
  - Made `output_dir` a required parameter in `run_yolo_inference()` (moved from last position with default to third position, no default)
  - Labels are now written directly to the output directory instead of being written in-place first and then copied
  - Original input directory remains completely untouched — no JSON files are created there
  - Updated docstring to reflect the new behavior
  - Rationale: Prevents polluting the source image directory with label files

**Git commit 86df169** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

## 2026-02-05

- **Autolabel output directory now derived from input directory**:
  - Modified files: `Pipeline/autolabel.py`, `Pipeline/workflow_config.json`, `Pipeline/yolo_training_workflow.py`, `CLAUDE.md`
  - Output dir is now a sibling `autolabeled/` folder next to the input dir's parent (e.g. `./extracted_frames/my_video` → `./autolabeled/my_video`)
  - Removed `autolabel_output_dir` config option from `workflow_config.json`, `DEFAULT_CONFIG`, and `CLAUDE.md` — no longer needed since the path is computed from the input
  - Rationale: Output naturally follows the input location instead of requiring a separate config key

- **Autolabel now copies labeled images and labels into a separate output folder**:
  - Modified files: `Pipeline/autolabel.py`, `Pipeline/workflow_config.json`, `Pipeline/yolo_training_workflow.py`, `CLAUDE.md`
  - Added `autolabel_output_dir` config option (default `./autolabeled`) — labeled images and their JSON labels are copied into this directory after inference
  - Added `output_dir` parameter to `run_yolo_inference()` — when set, each labeled image and its JSON file are copied to the output directory
  - Updated `yolo_autolabel()` to resolve `autolabel_output_dir` (appending video name as subdirectory when available) and pass it to inference
  - Updated config chaining: `config["labeling_input_dir"]` now points to the output directory so the next workflow step (anonymize) reads from it
  - Updated `_get_step_output_dir` in `yolo_training_workflow.py` to show autolabel output directory in workflow summary
  - Added `autolabel_output_dir` to `DEFAULT_CONFIG`, `workflow_config.json` (doc + value sections), and `CLAUDE.md`
  - Rationale: Keeps original extracted frames untouched; labeled results are isolated in their own folder for downstream steps

- **Removed dead config keys left over from deleted filtering step**:
  - Modified files: `Pipeline/workflow_config.json`, `Pipeline/extract_frames.py`, `Pipeline/anonymize.py`
  - Removed `filter_input_dir`, `kept_images_dir`, `deleted_images_dir`, `move_to_trash` from `workflow_config.json` (both documentation and value sections) — these belonged to the deleted `filter_images.py` step and were never read
  - Removed `review_output_dir` from `workflow_config.json` — no code reads this key (consolidate.py writes `review_input_dir`, a different name)
  - Removed dead `filter_input_dir` assignment from `extract_frames.py` (line 1203)
  - Updated `anonymize.py` fallback: replaced `kept_images_dir` reference with `anonymize_input_dir` (the filtering step no longer exists)
  - Changed `anonymize_input_dir` default from `./kept_images` to `./extracted_frames` in both `workflow_config.json` and `anonymize.py`
  - Renumbered documentation steps in `workflow_config.json` (removed "Step 2 - Image Filtering" gap)

- **Removed `train_model` and `yolo_model_path` config keys; always prompt for model selection**:
  - Modified files: `Pipeline/yolo_training_workflow.py`, `Pipeline/autolabel.py`, `Pipeline/workflow_config.json`
  - Removed `train_model` from `DEFAULT_CONFIG` and `workflow_config.json` (both doc and value sections); training fallback now hardcodes `yolov8n.pt` instead of reading from config
  - Removed `yolo_model_path` from `DEFAULT_CONFIG`, `workflow_config.json`, and `autolabel.py`; autolabel now always calls `select_yolo_model()` instead of skipping when config value was set
  - Removed `--model` CLI arg from `yolo_training_workflow.py` (it set `yolo_model_path`, which no longer exists)
  - Rationale: `select_yolo_model()` handles all model selection interactively; config overrides that bypass the prompt are no longer needed

- **Fixed YOLO runs directory path across all pipeline scripts**:
  - Modified files: `Pipeline/model_utils.py`, `Pipeline/autolabel.py`, `Pipeline/video_inference.py`, `Pipeline/yolo_training_workflow.py`
  - Changed `DEFAULT_RUNS_DIR` from `~/runs/detect/runs` to `/home/aidall/AI_Hub/runs/detect/runs` in all four files
  - Also updated the fallback default in `yolo_training_workflow.py` `train_yolo_model()` and `DEFAULT_CONFIG`
  - Rationale: The correct path is `/home/aidall/AI_Hub/runs/detect/runs`, not the current user's home directory

- **Fixed duplicate continuation prompt in YOLO workflow**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - When a step failed, the user was prompted "Continue to next step anyway?" and then immediately prompted again "Step X done. Continue to Step Y?" — a redundant double prompt
  - Restructured the logic so the failure path and success path each show a single, appropriate prompt: failure says "Continue to Step Y anyway?", success says "Step X done. Continue to Step Y?"
  - Rationale: Eliminates confusing duplicate prompt after step failures

- **Refactored YOLO model selection into shared interactive selector**:
  - New file: `Pipeline/model_utils.py`
  - Modified files: `Pipeline/autolabel.py`, `Pipeline/video_inference.py`, `Pipeline/yolo_training_workflow.py`
  - Created `model_utils.py` with `find_yolo_versions()` and `select_yolo_model()` — shared utilities for discovering and interactively selecting YOLO models
  - `select_yolo_model()` lists all `YOLO_v*` versions in the runs directory, suggests the latest as default (Enter to accept), lets user type a different version number (confirms before proceeding), or type `'c'` for a custom path
  - Removed duplicate `find_latest_yolo_model()` from `autolabel.py` and `video_inference.py`
  - `autolabel.py`: replaced inline model discovery + Y/n prompt with `select_yolo_model()` call
  - `video_inference.py`: replaced inline model discovery + Y/n prompt with `select_yolo_model()` call
  - `yolo_training_workflow.py`: replaced inline version scanning and model selection logic with `find_yolo_versions()` + `select_yolo_model()`; version number for new training run is now auto-incremented from highest existing version (no manual version prompt); base model selection uses the shared interactive selector
  - Rationale: Eliminates duplicated model discovery code, provides consistent interactive model selection UX across the pipeline

**Git commit c73209c** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

## 2026-02-05

- **Consolidated training model prompt into single question and fixed runs path**:
  - Modified files: `Pipeline/yolo_training_workflow.py`, `Pipeline/video_inference.py`, `Pipeline/autolabel.py`
  - Training Step 6: Combined model selection into one prompt — "Press Enter to use this model, or enter a version number"
  - If a version number is entered, looks up that version's `best.pt` and asks "Use this model?"
  - If version not found, falls back to manual path entry
  - Changed default `yolo_runs_dir` from `~/AI_Hub/runs/detect/runs` to `~/runs/detect/runs` in all three files (code runs inside AI_Hub)
  - Rationale: Single prompt is cleaner; path fix matches actual runtime directory

- **Added latest YOLO_v* model auto-detection to video inference and auto-labeling**:
  - Modified files: `Pipeline/video_inference.py`, `Pipeline/autolabel.py`
  - Both scripts now include `find_latest_yolo_model()` to scan `~/runs/detect/runs/` for the most recent `YOLO_v*` model
  - `video_inference.py`: `--model` is now optional; if omitted, finds latest model and prompts user to confirm before using it
  - `autolabel.py`: When no `yolo_model_path` in config, finds latest model and prompts user to confirm before using it
  - All three tools (training, inference, auto-labeling) now prompt the user to confirm the model choice
  - Rationale: Consistent model selection across the pipeline with user confirmation

- **Added YOLO versioning and auto model selection to Step 6 training**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Step 6 now scans `~/runs/detect/runs/` for existing `YOLO_v*` folders
  - Lists all existing versions and shows the most recent one
  - Prompts user for version number (default: next version, e.g. YOLO_v3)
  - Experiment name is now `YOLO_v<N>` instead of free-form text
  - Automatically uses `best.pt` from the most recent version as the base model for training
  - Falls back to config `train_model` (default `yolov8n.pt`) if no previous version exists
  - Training output saves to the runs directory (e.g. `~/AI_Hub/runs/detect/runs/YOLO_v3/`)
  - Added `yolo_runs_dir` config key (default: `~/AI_Hub/runs/detect/runs`)
  - Rationale: Consistent versioning scheme and transfer learning from the latest trained model

- **Added git commit separator rule to CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Added instruction to place git commit separators above the log entries they cover after committing
  - Documented the format: `**Git commit <short-hash>**...` followed by `---`
  - Rationale: Ensures consistent commit tracking in LOG.md

- **Added experiment name prompt to Step 6 YOLO training**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Step 6 now prompts the user to enter an experiment name before training starts
  - If user presses Enter, falls back to config `train_name` or auto-generates a timestamp-based name
  - Name is displayed in the training configuration summary
  - Rationale: Lets users name their training runs for easier identification

- **Made empty label cleanup search subdirectories recursively**:
  - Modified file: `Pipeline/labeling.py`
  - Changed `remove_empty_labels()` from `input_path.iterdir()` to `input_path.rglob("*")` so it finds images in subdirectories
  - Preserves subdirectory structure when moving files to `deleted/empty_labels/`
  - Rationale: Input directories may contain images organized in subfolders

- **Clarified log insertion rule in CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Updated Logging Policy to explicitly state that new entries must be added directly below the `# Change Log` heading and `---` separator, above all existing date sections
  - Added rule: if today's date section already exists at the top, append to it instead of creating a duplicate
  - Rationale: Previous wording ("add at the top") was ambiguous and caused inconsistent placement

- **Enhanced Step 6 dataset.yaml prompt with Step 5 output as default**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Step 6 now prompts user for dataset.yaml path instead of failing automatically
  - Uses Step 5 output directory as default (if dataset.yaml exists there)
  - If Step 5 was skipped or no dataset.yaml found, prompts user to enter path manually
  - Validates path exists before proceeding with training
  - Rationale: Allows users to specify custom dataset.yaml paths when Step 5 is skipped or use different datasets

- **Added per-frame FPS, GPU info, and FP16 acceleration to video inference**:
  - Modified file: `Pipeline/video_inference.py`
  - Batch mode now shows live per-frame FPS (smoothed) during processing, not just avg at end
  - Added `print_device_info()` — prints GPU name and memory at startup, or "CPU" if no CUDA
  - Added `--half` flag for FP16 half-precision inference (faster on GPUs with Tensor Cores)
  - Startup info now includes video duration and FP16 status
  - Rationale: Better speed visibility and GPU acceleration option

- **Created standalone YOLO video inference script with two modes**:
  - New file: `Pipeline/video_inference.py`
  - **Batch mode** (default): Processes video without display, saves annotated output MP4, shows progress and avg FPS
  - **Real-time mode** (`--show`): Opens OpenCV window with live bounding boxes, class labels, confidence scores, smoothed FPS counter, and detection count overlay. Press 'q' to quit
  - Both modes support: video files, webcam (`--source 0`), stream URLs, `.pt` and `.onnx` models
  - Optional `--save` flag in real-time mode to also write output video
  - Configurable confidence, IoU, image size, device, and class filtering via CLI args
  - Rationale: Standalone inference tool separate from the training pipeline

- **Changed default training results save location**:
  - Modified file: `Pipeline/train_yolo.py`
  - Changed default `project` parameter from `"."` to `"./YOLO_result"` (lines 164, 488)
  - Training results now save to `./YOLO_result/runs/train/<experiment_name>/` by default instead of `./runs/train/<experiment_name>/`
  - Rationale: Keeps training outputs organized in a dedicated directory separate from other pipeline outputs

---
**Git commit dad8a17** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

## 2026-02-04

- **Moved train_yolo.py from YOLO_Training to Pipeline directory**:
  - Moved file: `YOLO_Training/train_yolo.py` → `Pipeline/train_yolo.py`
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Removed YOLO_TRAINING_DIR path manipulation from workflow
  - Updated import to directly import from current directory (Pipeline)
  - Updated error message to reflect new location
  - Rationale: Consolidates all pipeline steps into Pipeline directory for better organization

- **Removed entire YOLO_Training directory and all files**:
  - Deleted files: `YOLO_Training/convert_json_to_yolo.py`, `YOLO_Training/args.yaml`, `YOLO_Training/best.onnx`, `YOLO_Training/convert_json_to_yolo.py:Zone.Identifier`, `YOLO_Training/x_anylabeling_config.yaml`, `YOLO_Training/dataset.yaml`
  - Rationale:
    - `convert_json_to_yolo.py` - Unused by workflow; conversion is handled by consolidate.py
    - `args.yaml` - Auto-generated training history record, not used as input
    - `best.onnx` - Optional reference model, not required for workflow
    - `convert_json_to_yolo.py:Zone.Identifier` - Windows security metadata file
    - `x_anylabeling_config.yaml` - Misplaced config file not used in YOLO training
    - `dataset.yaml` - Template file; workflow auto-generates dataset.yaml in consolidated output directory
    - `train_yolo.py` - Moved to Pipeline directory (see above)
  - YOLO_Training directory is now empty and can be removed

- **Enhanced logging policy in CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Added explicit reminder to record changes to LOG.md immediately after code modifications
  - Emphasized that LOG.md updates should happen before git commits
  - Rationale: Ensures no changes are forgotten in the changelog

- **Removed unused convert_json_to_yolo import from yolo_training_workflow.py**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Removed unused import: `from convert_json_to_yolo import convert_dataset`
  - Removed `convert_dataset = None` fallback variable
  - Updated error message to only mention train_yolo.py requirement
  - Rationale: YOLO conversion is handled by consolidate.py's built-in code (_convert_to_yolo_format function), not the standalone convert_json_to_yolo.py module. The import was never used in the workflow.

---
**Git commit cdf768f** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
---

## 2026-02-04

- **Updated logging policy in CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Added explicit instruction to never modify past log entries
  - Clarified that only new entries should be added at the top of LOG.md
  - Ensures historical logs remain unchanged to maintain accurate project history

- **Renamed yolo_training_workflow.py (formerly yolo_labeling_workflow.py)**:
  - Renamed file: `Pipeline/yolo_labeling_workflow.py` → `Pipeline/yolo_training_workflow.py`
  - Modified files: `Pipeline/yolo_training_workflow.py`, `Pipeline/workflow_config.json`, `CLAUDE.md`, `WORKFLOW_FEATURES.md`, `LOG.md`
  - Updated all command-line usage examples in yolo_training_workflow.py docstrings
  - Updated workflow banner from "YOLO AUTO-LABELING WORKFLOW" to "YOLO TRAINING WORKFLOW"
  - Updated argparse description to emphasize training aspect
  - Updated workflow title in CLAUDE.md from "YOLO Auto-Labeling Workflow" to "YOLO Training Workflow"
  - Updated section headers in CLAUDE.md and workflow_config.json
  - Updated WORKFLOW_FEATURES.md title and references
  - Updated all historical LOG.md references to use new name
  - Rationale: Better reflects the complete workflow which includes training as the final step, not just auto-labeling

- **Renamed yolo_autolabel.py to autolabel.py**:
  - Renamed file: `Pipeline/yolo_autolabel.py` → `Pipeline/autolabel.py`
  - Modified files: `Pipeline/yolo_training_workflow.py`, `Pipeline/anonymize.py`, `CLAUDE.md`, `WORKFLOW_FEATURES.md`, `LOG.md`
  - Updated import statement in yolo_training_workflow.py: `from autolabel import yolo_autolabel`
  - Updated comment in anonymize.py referencing the autolabel module
  - Updated file reference in CLAUDE.md project structure
  - Updated file reference in WORKFLOW_FEATURES.md Step 2 documentation
  - Replaced all historical references to yolo_autolabel with autolabel throughout LOG.md
  - Rationale: Simplified naming to match other step modules (extract_frames, consolidate, etc.)

## 2026-02-04

- **Added YOLO training parameters to workflow_config.json**:
  - Modified files: `Pipeline/workflow_config.json`, `CLAUDE.md`
  - Added 16 training configuration parameters to workflow_config.json:
    - `train_model`, `train_epochs`, `train_batch`, `train_imgsz`, `train_device`
    - `train_workers`, `train_project`, `train_name`, `train_resume`, `train_pretrained`
    - `train_optimizer`, `train_lr0`, `train_patience`, `train_cache`, `train_amp`, `train_augment`
  - Added documentation section explaining each parameter with defaults
  - Training parameters can now be customized in workflow_config.json instead of editing Python code
  - Workflow falls back to DEFAULT_CONFIG in yolo_training_workflow.py if parameters are missing
  - Updated CLAUDE.md to document that training config is now in workflow_config.json
  - Clarified that DEFAULT_CONFIG serves as fallback defaults

- **Corrected YOLO training configuration documentation in CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Rewrote "Key Configuration" section to accurately reflect where YOLO training parameters are stored
  - Clarified that `workflow_config.json` does NOT contain YOLO training parameters (NOTE: This was corrected later - see above entry)
  - Documented that training config for integrated workflow (Step 6) is in `yolo_training_workflow.py` DEFAULT_CONFIG dictionary (lines 115-132)
  - Added list of all 17 training parameters available in DEFAULT_CONFIG
  - Documented that `args.yaml` is auto-generated by Ultralytics as a record, NOT used as input configuration
  - Explained that standalone `train_yolo.py` uses command-line arguments, not args.yaml
  - Added instructions for customizing training parameters (edit DEFAULT_CONFIG or use custom JSON config file)
  - Removed incorrect claim that workflow_config.json contains parameters like `train_model`, `train_epochs`, etc.

## 2026-02-04

- **Removed all temporary debugging code from pipeline**:
  - Modified files: `Pipeline/autolabel.py`, `Pipeline/extract_frames.py`
  - Removed GPU detection and device verification code from `autolabel.py` (lines 167-193)
  - Removed timing instrumentation from `extract_frames.py`:
    - Removed timing variable declarations (timing_parallel, timing_cluster, timing_copy)
    - Removed all `t0 = time.time()` and timing accumulation statements
    - Removed timing breakdown output section that printed performance metrics
  - All temporary code marked with TODO comments has been cleaned up

- **Added Step 6 (Train YOLO model) to YOLO training workflow**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Added new step 6: "Train YOLO model" after YOLO format conversion
  - Imported training utilities from `YOLO_Training/train_yolo.py` and `convert_json_to_yolo.py`
  - Added `train_yolo_model()` function that:
    - Validates `dataset.yaml` exists from step 5
    - Extracts training parameters from config (model, epochs, batch, imgsz, device, etc.)
    - Calls `train_yolo()` with all parameters
    - Stores training results directory in config
  - Added 17 new training configuration parameters to `DEFAULT_CONFIG`:
    - `train_model`, `train_epochs`, `train_batch`, `train_imgsz`, `train_device`
    - `train_workers`, `train_project`, `train_name`, `train_resume`, `train_pretrained`
    - `train_optimizer`, `train_lr0`, `train_patience`, `train_cache`, `train_amp`, `train_augment`
  - Updated workflow to support 6 steps instead of 5:
    - `print_step_menu()`: Added step 6 to menu
    - `get_step_choice()`: Changed range from 1-5 to 1-6
    - `run_workflow()`: Added step 6 to steps list, updated loop conditions
    - `_get_step_output_dir()`: Added mapping for step 6 → `training_results_dir`
    - Argument parser: Changed `--start-step` choices to include 6
  - Updated documentation strings and epilog to list 6 steps
  - Added `YELLOW` color constant for training configuration display
  - Training function displays configuration before starting and prompts confirmation
  - Workflow continues to summary display after training completes or fails

- **Integrated YOLO conversion directly into consolidate.py**:
  - Modified files: `Pipeline/consolidate.py`, `Pipeline/yolo_training_workflow.py`
  - Moved all YOLO conversion logic from `YOLO_Training/convert_json_to_yolo.py` into `consolidate.py`
  - Added new functions to `consolidate.py`:
    - `_extract_annotations_from_json()`: Extract bounding boxes from JSON
    - `_convert_box_to_yolo()`: Convert coordinates to YOLO normalized format
    - `_build_class_mapping()`: Scan JSON files to extract unique classes
    - `_process_json_to_yolo()`: Process single JSON file to YOLO TXT
    - `_convert_to_yolo_format()`: Main conversion function (train/val split, YAML generation)
  - Modified `consolidate_files()` to call `_convert_to_yolo_format()` when `config["convert_to_yolo"] = True`
  - Removed import of `convert_json_to_yolo` from `yolo_training_workflow.py`
  - Simplified `consolidate_and_convert()` in workflow - now just sets config flags and calls `consolidate_files()`
  - YOLO conversion now happens internally within consolidation step, no external dependencies

- **Removed Step 4 from YOLO training workflow**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Removed the first consolidation step (Step 4: "Consolidate files")
  - Workflow now goes directly from Anonymize (Step 3) → Review/Correct Labels (Step 4) → Final Consolidate & Convert (Step 5)
  - X-AnyLabeling now opens the anonymized images directory directly without pre-consolidation
  - Eliminates unnecessary file copying and renaming before manual review
  - Updated step numbers throughout: menu, progress display, argument parser, and help text
  - Renamed `consolidate_together()` function (removed, was unused)
  - Renamed `consolidate_separated()` to `consolidate_and_convert()`
  - Updated `_get_step_output_dir()` mapping to reflect new 5-step structure
  - Workflow now has 5 steps instead of 6

- **Created comprehensive feature documentation**:
  - Created new file: `WORKFLOW_FEATURES.md`
  - Documented all features for each step of the YOLO training workflow
  - Includes step-by-step breakdowns with configuration keys, examples, and troubleshooting
  - Covers input options, deduplication methods, quality filtering, GPU acceleration, format detection, overwrite protection, interactive configuration, error handling, and more
  - Added command-line usage guide, tips & best practices, and output directory structure
  - Comprehensive reference document for understanding workflow capabilities

- **Added temporary GPU detection to autolabel.py**:
  - Modified file: `Pipeline/autolabel.py`
  - Added GPU availability check and device detection before model inference
  - Prints GPU name, memory, and whether model is using GPU or CPU
  - Temporary debugging code marked with TODO comments for easy removal

## 2026-02-03

- **— Fixed convert_dataset() search_dirs argument error**:
  - Modified file: `Pipeline/yolo_training_workflow.py`
  - Removed unused `search_dirs` variable and its argument from the `convert_dataset()` call in `consolidate_separated()`
  - The `search_dirs` parameter was being passed but `convert_dataset()` in `convert_json_to_yolo.py` does not accept it
  - Error was: `convert_dataset() got an unexpected keyword argument 'search_dirs'`

- **— Added UI Guidelines section to CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Added requirement that all warning messages displayed to users must be colored red (ANSI escape code `\033[91m`)

- **— Added automatic classes.txt lookup for YOLO conversion**:
  - Modified files: `YOLO_Training/convert_json_to_yolo.py`, `Pipeline/consolidate.py`, `Pipeline/yolo_training_workflow.py`
  - When merging multiple datasets, the YOLO conversion now searches input directories for `classes.txt` files and uses the one with the most classes
  - This ensures class IDs remain consistent when combining datasets with different numbers of classes
  - Added `find_largest_classes_file()` function to scan directories for classes.txt
  - Updated `load_classes_from_file()` to return a list of class names (supports both plain and numbered formats)
  - Added `search_dirs` parameter to `build_class_mapping()` and `convert_dataset()`
  - `consolidate.py` now stores `consolidate_input_dirs` in config for YOLO class lookup
  - Priority: explicit classes_file > largest classes.txt from input dirs > auto-detect from JSON

- **— Parallelized image processing with multiprocessing**:
  - Modified file: `extract_frames.py`
  - Added multiprocessing Pool to process images in parallel across CPU cores
  - Image loading, blur detection, and histogram computation now run in parallel
  - Uses `min(cpu_count(), 8)` workers to avoid excessive overhead
  - Added `_init_worker()` and `_process_single_image()` functions for parallel processing
  - Updated timing output to show combined parallel processing time

- **— Added timing instrumentation and removed deleted folder copying**:
  - Modified file: `extract_frames.py`
  - Added temporary timing output to identify performance bottlenecks (image loading, blur detection, histogram, clustering, file copying)
  - Removed copying of blurry and non-representative images to deleted folder — they are now skipped entirely
  - This reduces I/O operations significantly (only copies representatives to output)
  - Timing output marked with TODO comments for easy removal later

- **— Added sequential clustering method as faster alternative to DBSCAN**:
  - Modified files: `extract_frames.py`, `workflow_config.json`
  - Added `cluster_sequential_frames()` function using sliding window + Union-Find algorithm
  - O(n × window_size) complexity instead of O(n²) for DBSCAN — much faster for video frames
  - Sequential method compares each frame only to its neighboring frames (default window=10)
  - Added `prompt_clustering_method()` to let user choose: 1=Sequential (fast), 2=DBSCAN (thorough)
  - Added config options: `clustering_method` (null=prompt, 'sequential', 'dbscan') and `clustering_window_size` (default 10)

- **— Lowered blur_threshold from 100.0 to 80.0**:
  - Modified file: `workflow_config.json`
  - Reduced `blur_threshold` to keep more images that were previously filtered as blurry
  - Folders where all images fell below the old threshold were silently skipped during clustering

- **— Updated CLAUDE.md logging policy**:
  - Modified file: `CLAUDE.md`
  - Updated the Logging Policy section to require explicit listing of modified code files by filename
  - Changed the explanation format from prose to bullet points

- **— Fixed NameError in consolidate.py line 436**: `detect_label_format(input_dir)` referenced undefined variable `input_dir`. Changed to `first_input`, which is the directory path collected from the user prompt on line 403. The variable `input_dirs` (plural, a list) was defined on line 416 but `input_dir` (singular) was never defined.
- **— Added workflow summary printed after completion/stop**: Both `image_labeling_workflow.py` and `yolo_training_workflow.py` now print a summary when the workflow ends (whether all steps complete, user stops early, or a step fails). Shows each step that ran with a checkmark or X for success/failure, plus the output directory for each step. The YOLO workflow also shows the `dataset.yaml` path if YOLO conversion was performed.
- **— Fixed progress block showing skipped steps as completed**: `print_progress()` now takes a `start_step` parameter to distinguish skipped steps (before the user's chosen start step) from actually completed steps. Skipped steps show a dimmed dash (`─`), completed steps show a green checkmark (`✔`), active step shows a yellow arrow (`▶`), and pending steps show a dimmed circle (`○`).
- **— Replaced step headers with progress blocks in both workflows**: `run_workflow()` in `image_labeling_workflow.py` and `yolo_training_workflow.py` now prints a compact progress block before each step showing all steps with icons. Replaces the old `# STEP N: ...` headers and multi-line completion banners. The "continue to next step?" prompt is condensed to a single line. Also changed `consolidate.py` per-file logging (`[idx/total] ...`) to overwrite in-place using `\r\033[K` instead of printing a new line for each batch.
- **— Added YOLO conversion indicator to Step 6 confirmation prompt**: The consolidation confirmation now shows `JSON (LabelMe/X-AnyLabeling) -> YOLO TXT (auto-converted)` in the Labels line when called from `yolo_training_workflow.py`, so the user knows labels will be converted.
- **— Added multiple input directory support to consolidate_files**: The consolidation confirmation prompt now supports option `3` to add additional input directories. Multiple directories are consolidated sequentially into the same output with continuous numbering. Option `1` supports replacing or removing individual directories (via `r` prefix, e.g. `r2`) when multiple are listed.

## 2026-01-31

- **— Added YOLO format conversion to Step 6 of yolo_training_workflow.py**: After the final consolidation (Step 6), the workflow now automatically converts JSON labels to YOLO txt format by calling `convert_dataset()` from `YOLO_Training/convert_json_to_yolo.py`. The conversion uses the same output directory the user chose during consolidation (no extra prompts), producing `images/train`, `images/val`, `labels/train`, `labels/val` with a `dataset.yaml` ready for `train_yolo.py`. The intermediate `Image/` and `Label/` folders are removed after conversion so only the YOLO structure remains. Added config keys `yolo_train_ratio` and `yolo_classes_file` to `workflow_config.json`.
- **— Fixed stale YOLO output causing wrong numbering on repeated runs**: When running Step 6 into a directory that already contained output from a previous conversion, the old `images/`, `labels/`, `classes.txt`, and `dataset.yaml` would persist. `convert_dataset` would then mix old and new files together, producing incorrect train/val splits with mismatched image-label pairs. Fix: `consolidate_separated` now removes any existing `images/`, `labels/`, `classes.txt`, and `dataset.yaml` in the output directory before running the YOLO conversion, so each run starts clean.
- **— Made Step 6 skip the label format prompt**: Step 6 in `yolo_training_workflow.py` now sets `skip_format_prompt=True` so `consolidate_files` uses the configured `label_format` (default: `json`) directly instead of prompting. The `label_format` is configurable via `workflow_config.json`. Added `skip_format_prompt` support to `consolidate.py`'s `consolidate_files` function.
- **— Changed `cluster_image_directory` to copy instead of move**: Blurry and non-representative images are now copied to the deleted directory (`shutil.copy2`) instead of moved (`shutil.move`). The input directory is left untouched — previously it was partially emptied, leaving only representative images behind (duplicating what was already in `extracted_frames`).
- **— Fixed labeling.py input directory and empty label cleanup**: Fixed `run_labeling()` to use `labeling_input_dir` directly (without appending `video_name`) when it was explicitly set by a previous step like consolidate — previously it always appended `video_name`, causing "Directory not found" errors for `./Dataset/L(~)`. Removed the "Images to label" print line. Changed `_prompt_remove_empty()` to ask the user which directory to check instead of automatically using the labeling input directory.
- **— Fixed anonymize.py input directory for YOLO training workflow**: When `from_previous_step=True`, `anonymize_images()` now checks `labeling_input_dir` (set by `autolabel`) before falling back to `kept_images_dir/{video_name}`. Previously it always used `kept_images_dir`, which doesn't exist in the YOLO training workflow since there's no filter step — causing "Input directory does not exist" errors.
- **— Per-folder clustering in `cluster_image_directory`**: Restructured `cluster_image_directory()` to group images by parent folder and cluster each folder independently, rather than building one O(n²) distance matrix across the entire dataset. Images from different subdirectories (likely different video sources) are never compared against each other. Also prints per-folder progress when multiple folders are found.
- **— Made `find_image_files` recursive for subdirectory support**: Changed `find_image_files()` in `extract_frames.py` from non-recursive (`*.{ext}`) to recursive (`**/*.{ext}`) glob patterns, matching the behavior of `find_video_files()`. This means `cluster_image_directory()` now discovers images in subdirectories. Also updated `cluster_image_directory()` to preserve relative subdirectory paths when moving/copying files to output and deleted directories, preventing filename collisions from different subdirectories.
- **— Unified all deleted images under `./deleted/` directory**: All image/label deletion across the pipeline now moves files to subdirectories under `./deleted/` instead of separate directories or permanent deletion. Subdirectories: `deleted/clustering` (Step 1 — blurry + non-representative), `deleted/filtered` (Step 2 — manually rejected), `deleted/empty_labels` (Step 4 — images with no annotations), `deleted/unlabeled` (YOLO — images with no detections). Updated `labeling.py` to move instead of `unlink()`. Updated `autolabel.py` to always move instead of conditionally deleting. Updated all config defaults in `image_labeling_workflow.py`, `yolo_training_workflow.py`, and `workflow_config.json`.
- **— Extended Step 1 to support image directory clustering**: Step 1 (`extract_frames.py`) now handles image directories in addition to videos. When given a directory with no video files, it applies DBSCAN clustering + blur detection to deduplicate the images, keeping the sharpest representative per cluster and moving non-representatives/blurry images to a deleted directory. Video inputs now always use clustering mode. Added `cluster_image_directory()` and `find_image_files()` functions. Added `clustering_deleted_dir` config key. Updated `workflow_config.json` documentation. Removed the `clustering` boolean toggle (clustering is now always on).
- **— Added empty label cleanup option to labeling.py**: After X-AnyLabeling closes, the user is prompted to remove images with empty or missing labels. Handles all three label formats (JSON with empty `shapes`, empty TXT, XML with no `object` elements). Removes both the image and its label file. Works in both the auto-launch and manual-labeling code paths.
- **— Added red ANSI color for warnings/errors across all Pipeline scripts**: Added `RED = "\033[91m"` to all Pipeline files (`extract_frames.py`, `filter_images.py`, `anonymize.py`, `labeling.py`, `consolidate.py`, `review_labels.py`, `autolabel.py`, `image_labeling_workflow.py`, `yolo_training_workflow.py`). All error messages, warnings, and "not found" messages now display in red for visibility.
- **— Updated autolabel.py to four-point rectangle mode**: Changed `create_labelme_json` to output bounding boxes as four corner points (top-left, top-right, bottom-right, bottom-left) instead of two diagonal points. Fixes X-AnyLabeling v2.2.0+ deprecation warning: "Diagonal vertex mode is deprecated."
- **— Removed duplicate step headings from all step files**: Each step file (step1 through step6, plus step_autolabel) printed its own "STEP N: ..." banner, duplicating the heading already printed by the workflow orchestrators. Removed the banners from all step files so only the workflow prints the heading.
- **— Removed duplicate heading from step_autolabel.py**: The "STEP: YOLO AUTO-LABELING" banner was printed by both `yolo_training_workflow.py` and `step_autolabel.py`. Removed the one from the step file since the workflow already prints it.
- **— Added green-colored prompts to all Pipeline scripts**: Added ANSI green color (`\033[92m`) to all `input()` prompts across `image_labeling_workflow.py`, `yolo_training_workflow.py`, `step1_extract_frames.py`, `step2_filter_images.py`, `step3_anonymize.py`, `step4_labeling.py`, `step5_consolidate.py`, `step6_review_labels.py`, and `step_autolabel.py`. Questions/prompts that require user input now display in green for better visibility.
- **— Updated step_autolabel.py**: Changed image file discovery from `iterdir()` to `rglob('*')` so YOLO auto-labeling searches subdirectories recursively, not just the top-level input directory.
- **— Fixed step_autolabel.py**: Pass list of image file paths to `model.predict()` instead of the directory path. Fixes Ultralytics failing on directories with Korean characters or parentheses in the path.
- **— Fixed step_autolabel.py**: Process images in batches of 500 instead of all at once to avoid "too many open files" OS error on large datasets.
- **— Fixed step_autolabel.py**: Use batch size of 1 for ONNX models (fixed batch dimension) vs 500 for .pt models. Fixes `INVALID_ARGUMENT: Got: 500 Expected: 1` error.
- **— Removed duplicate config prints from step_autolabel.py**: Model/Confidence/IoU/Image size/Delete unlabeled were printed twice — once in `run_autolabel_step()` and again in `run_yolo_inference()`. Removed the duplicates from `run_yolo_inference()`.
- **— Removed duplicate step list from yolo_training_workflow.py banner**: The numbered step list appeared in both the banner and the step selection menu. Removed it from the banner so it only shows once in the menu.
- **— Removed duplicate label format listing from step5_consolidate.py**: Label file counts were shown twice — once as a summary and again in the selection menu. Removed the standalone summary so counts only appear in the menu.
- **— Fixed step4_labeling.py Qt plugin crash when run from yolo_env**: X-AnyLabeling failed with "Could not load the Qt platform plugin xcb" because `subprocess.Popen` inherited the full yolo_env environment. Shell-level unsets weren't enough. Now passes a clean `env` dict to Popen with only essential system variables (HOME, DISPLAY, etc.) and a PATH stripped of all venv entries, so the x-anylabeling_env activate script starts from a clean slate.
- **— Removed input directory requirement from step4_labeling.py**: X-AnyLabeling doesn't need the input directory passed to it. Removed the existence check and interactive prompt that blocked if the directory wasn't found. The directory path is still resolved and printed as a hint.
- **— Renamed all step files to remove stepX_ prefix**: `step1_extract_frames.py` → `extract_frames.py`, `step2_filter_images.py` → `filter_images.py`, `step3_anonymize.py` → `anonymize.py`, `step4_labeling.py` → `labeling.py`, `step5_consolidate.py` → `consolidate.py`, `step6_review_labels.py` → `review_labels.py`, `step_autolabel.py` → `autolabel.py`. Updated imports in `image_labeling_workflow.py`, `yolo_training_workflow.py`, and `CLAUDE.md`.
- **— Reordered yolo_training_workflow.py steps**: New order is: 1) Extract frames, 2) YOLO auto-label, 3) Anonymize, 4) Consolidate, 5) Review/correct labels (X-AnyLabeling), 6) Consolidate (final). Consolidation runs twice — before and after manual label review.
- **— Added existing output directory warning to consolidate.py**: Before proceeding, warns the user if the output directory already has files, showing the total file count and image count.
- **— Fixed consolidate output format for X-AnyLabeling compatibility in yolo workflow**: Step 4 (consolidate before review) now keeps images and labels together in the same folder (`separate_folders=False`) so X-AnyLabeling can find them. Step 6 (final consolidate) separates into Image/Label subfolders. Added `separate_folders` config key to `consolidate.py`. Also sets `labeling_input_dir` after step 4 so step 5 (X-AnyLabeling) shows the correct directory hint.

## 2026-01-30

- **Updated workflow_config.json**: Added YOLO auto-labeling config values (yolo_model_path, autolabel_confidence, autolabel_iou, autolabel_imgsz, autolabel_device, autolabel_delete_unlabeled, autolabel_deleted_dir) with documentation, so both workflows share one config file.
- **Created YOLO training workflow**: Added `Pipeline/yolo_training_workflow.py` (orchestrator) and `Pipeline/step_autolabel.py` (YOLO inference step). This is a separate 6-step pipeline that uses a trained YOLO model to auto-label extracted frames, deletes images with no detections, anonymizes, opens X-AnyLabeling for review/correction, consolidates the dataset, and trains a YOLO model. Reuses existing step1, step3, step4, and step5 modules.
- **Fixed image_labeling_workflow.py**: Changed `from steps import ...` to direct module imports (`from step1_extract_frames import ...`, etc.) so the script works when run from the Pipeline directory. The previous import referenced a nonexistent `steps` package.
- **Updated CLAUDE.md**: Added note that LOG.md writes are automatic and do not require user permission.
- **Updated CLAUDE.md**: Updated logging policy to require time in addition to date for each log entry.
- **Updated LOG.md**: Reformatted existing entries to include timestamps.
- **Created CLAUDE.md**: Added project documentation covering structure, data flow, setup, pipeline usage, training, annotation formats, and key configuration.
- **Created LOG.md**: Added this change log to track all modifications made to the project.
- **Updated CLAUDE.md**: Added logging policy requiring all changes to be documented in LOG.md.
