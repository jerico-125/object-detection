# Change Log

## 2026-02-04

- **— Added temporary GPU detection to yolo_autolabel.py**:
  - Modified file: `Pipeline/yolo_autolabel.py`
  - Added GPU availability check and device detection before model inference
  - Prints GPU name, memory, and whether model is using GPU or CPU
  - Temporary debugging code marked with TODO comments for easy removal

## 2026-02-03

- **— Fixed convert_dataset() search_dirs argument error**:
  - Modified file: `Pipeline/yolo_labeling_workflow.py`
  - Removed unused `search_dirs` variable and its argument from the `convert_dataset()` call in `consolidate_separated()`
  - The `search_dirs` parameter was being passed but `convert_dataset()` in `convert_json_to_yolo.py` does not accept it
  - Error was: `convert_dataset() got an unexpected keyword argument 'search_dirs'`

- **— Added UI Guidelines section to CLAUDE.md**:
  - Modified file: `CLAUDE.md`
  - Added requirement that all warning messages displayed to users must be colored red (ANSI escape code `\033[91m`)

- **— Added automatic classes.txt lookup for YOLO conversion**:
  - Modified files: `YOLO_Training/convert_json_to_yolo.py`, `Pipeline/consolidate.py`, `Pipeline/yolo_labeling_workflow.py`
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
- **— Added workflow summary printed after completion/stop**: Both `image_labeling_workflow.py` and `yolo_labeling_workflow.py` now print a summary when the workflow ends (whether all steps complete, user stops early, or a step fails). Shows each step that ran with a checkmark or X for success/failure, plus the output directory for each step. The YOLO workflow also shows the `dataset.yaml` path if YOLO conversion was performed.
- **— Fixed progress block showing skipped steps as completed**: `print_progress()` now takes a `start_step` parameter to distinguish skipped steps (before the user's chosen start step) from actually completed steps. Skipped steps show a dimmed dash (`─`), completed steps show a green checkmark (`✔`), active step shows a yellow arrow (`▶`), and pending steps show a dimmed circle (`○`).
- **— Replaced step headers with progress blocks in both workflows**: `run_workflow()` in `image_labeling_workflow.py` and `yolo_labeling_workflow.py` now prints a compact progress block before each step showing all steps with icons. Replaces the old `# STEP N: ...` headers and multi-line completion banners. The "continue to next step?" prompt is condensed to a single line. Also changed `consolidate.py` per-file logging (`[idx/total] ...`) to overwrite in-place using `\r\033[K` instead of printing a new line for each batch.
- **— Added YOLO conversion indicator to Step 6 confirmation prompt**: The consolidation confirmation now shows `JSON (LabelMe/X-AnyLabeling) -> YOLO TXT (auto-converted)` in the Labels line when called from `yolo_labeling_workflow.py`, so the user knows labels will be converted.
- **— Added multiple input directory support to consolidate_files**: The consolidation confirmation prompt now supports option `3` to add additional input directories. Multiple directories are consolidated sequentially into the same output with continuous numbering. Option `1` supports replacing or removing individual directories (via `r` prefix, e.g. `r2`) when multiple are listed.

## 2026-01-31

- **— Added YOLO format conversion to Step 6 of yolo_labeling_workflow.py**: After the final consolidation (Step 6), the workflow now automatically converts JSON labels to YOLO txt format by calling `convert_dataset()` from `YOLO_Training/convert_json_to_yolo.py`. The conversion uses the same output directory the user chose during consolidation (no extra prompts), producing `images/train`, `images/val`, `labels/train`, `labels/val` with a `dataset.yaml` ready for `train_yolo.py`. The intermediate `Image/` and `Label/` folders are removed after conversion so only the YOLO structure remains. Added config keys `yolo_train_ratio` and `yolo_classes_file` to `workflow_config.json`.
- **— Fixed stale YOLO output causing wrong numbering on repeated runs**: When running Step 6 into a directory that already contained output from a previous conversion, the old `images/`, `labels/`, `classes.txt`, and `dataset.yaml` would persist. `convert_dataset` would then mix old and new files together, producing incorrect train/val splits with mismatched image-label pairs. Fix: `consolidate_separated` now removes any existing `images/`, `labels/`, `classes.txt`, and `dataset.yaml` in the output directory before running the YOLO conversion, so each run starts clean.
- **— Made Step 6 skip the label format prompt**: Step 6 in `yolo_labeling_workflow.py` now sets `skip_format_prompt=True` so `consolidate_files` uses the configured `label_format` (default: `json`) directly instead of prompting. The `label_format` is configurable via `workflow_config.json`. Added `skip_format_prompt` support to `consolidate.py`'s `consolidate_files` function.
- **— Changed `cluster_image_directory` to copy instead of move**: Blurry and non-representative images are now copied to the deleted directory (`shutil.copy2`) instead of moved (`shutil.move`). The input directory is left untouched — previously it was partially emptied, leaving only representative images behind (duplicating what was already in `extracted_frames`).
- **— Fixed labeling.py input directory and empty label cleanup**: Fixed `run_labeling()` to use `labeling_input_dir` directly (without appending `video_name`) when it was explicitly set by a previous step like consolidate — previously it always appended `video_name`, causing "Directory not found" errors for `./Dataset/L(~)`. Removed the "Images to label" print line. Changed `_prompt_remove_empty()` to ask the user which directory to check instead of automatically using the labeling input directory.
- **— Fixed anonymize.py input directory for YOLO workflow**: When `from_previous_step=True`, `anonymize_images()` now checks `labeling_input_dir` (set by `yolo_autolabel`) before falling back to `kept_images_dir/{video_name}`. Previously it always used `kept_images_dir`, which doesn't exist in the YOLO workflow since there's no filter step — causing "Input directory does not exist" errors.
- **— Per-folder clustering in `cluster_image_directory`**: Restructured `cluster_image_directory()` to group images by parent folder and cluster each folder independently, rather than building one O(n²) distance matrix across the entire dataset. Images from different subdirectories (likely different video sources) are never compared against each other. Also prints per-folder progress when multiple folders are found.
- **— Made `find_image_files` recursive for subdirectory support**: Changed `find_image_files()` in `extract_frames.py` from non-recursive (`*.{ext}`) to recursive (`**/*.{ext}`) glob patterns, matching the behavior of `find_video_files()`. This means `cluster_image_directory()` now discovers images in subdirectories. Also updated `cluster_image_directory()` to preserve relative subdirectory paths when moving/copying files to output and deleted directories, preventing filename collisions from different subdirectories.
- **— Unified all deleted images under `./deleted/` directory**: All image/label deletion across the pipeline now moves files to subdirectories under `./deleted/` instead of separate directories or permanent deletion. Subdirectories: `deleted/clustering` (Step 1 — blurry + non-representative), `deleted/filtered` (Step 2 — manually rejected), `deleted/empty_labels` (Step 4 — images with no annotations), `deleted/unlabeled` (YOLO — images with no detections). Updated `labeling.py` to move instead of `unlink()`. Updated `yolo_autolabel.py` to always move instead of conditionally deleting. Updated all config defaults in `image_labeling_workflow.py`, `yolo_labeling_workflow.py`, and `workflow_config.json`.
- **— Extended Step 1 to support image directory clustering**: Step 1 (`extract_frames.py`) now handles image directories in addition to videos. When given a directory with no video files, it applies DBSCAN clustering + blur detection to deduplicate the images, keeping the sharpest representative per cluster and moving non-representatives/blurry images to a deleted directory. Video inputs now always use clustering mode. Added `cluster_image_directory()` and `find_image_files()` functions. Added `clustering_deleted_dir` config key. Updated `workflow_config.json` documentation. Removed the `clustering` boolean toggle (clustering is now always on).
- **— Added empty label cleanup option to labeling.py**: After X-AnyLabeling closes, the user is prompted to remove images with empty or missing labels. Handles all three label formats (JSON with empty `shapes`, empty TXT, XML with no `object` elements). Removes both the image and its label file. Works in both the auto-launch and manual-labeling code paths.
- **— Added red ANSI color for warnings/errors across all Pipeline scripts**: Added `RED = "\033[91m"` to all Pipeline files (`extract_frames.py`, `filter_images.py`, `anonymize.py`, `labeling.py`, `consolidate.py`, `review_labels.py`, `yolo_autolabel.py`, `image_labeling_workflow.py`, `yolo_labeling_workflow.py`). All error messages, warnings, and "not found" messages now display in red for visibility.
- **— Updated yolo_autolabel.py to four-point rectangle mode**: Changed `create_labelme_json` to output bounding boxes as four corner points (top-left, top-right, bottom-right, bottom-left) instead of two diagonal points. Fixes X-AnyLabeling v2.2.0+ deprecation warning: "Diagonal vertex mode is deprecated."
- **— Removed duplicate step headings from all step files**: Each step file (step1 through step6, plus step_yolo_autolabel) printed its own "STEP N: ..." banner, duplicating the heading already printed by the workflow orchestrators. Removed the banners from all step files so only the workflow prints the heading.
- **— Removed duplicate heading from step_yolo_autolabel.py**: The "STEP: YOLO AUTO-LABELING" banner was printed by both `yolo_labeling_workflow.py` and `step_yolo_autolabel.py`. Removed the one from the step file since the workflow already prints it.
- **— Added green-colored prompts to all Pipeline scripts**: Added ANSI green color (`\033[92m`) to all `input()` prompts across `image_labeling_workflow.py`, `yolo_labeling_workflow.py`, `step1_extract_frames.py`, `step2_filter_images.py`, `step3_anonymize.py`, `step4_labeling.py`, `step5_consolidate.py`, `step6_review_labels.py`, and `step_yolo_autolabel.py`. Questions/prompts that require user input now display in green for better visibility.
- **— Updated step_yolo_autolabel.py**: Changed image file discovery from `iterdir()` to `rglob('*')` so YOLO auto-labeling searches subdirectories recursively, not just the top-level input directory.
- **— Fixed step_yolo_autolabel.py**: Pass list of image file paths to `model.predict()` instead of the directory path. Fixes Ultralytics failing on directories with Korean characters or parentheses in the path.
- **— Fixed step_yolo_autolabel.py**: Process images in batches of 500 instead of all at once to avoid "too many open files" OS error on large datasets.
- **— Fixed step_yolo_autolabel.py**: Use batch size of 1 for ONNX models (fixed batch dimension) vs 500 for .pt models. Fixes `INVALID_ARGUMENT: Got: 500 Expected: 1` error.
- **— Removed duplicate config prints from step_yolo_autolabel.py**: Model/Confidence/IoU/Image size/Delete unlabeled were printed twice — once in `run_autolabel_step()` and again in `run_yolo_inference()`. Removed the duplicates from `run_yolo_inference()`.
- **— Removed duplicate step list from yolo_labeling_workflow.py banner**: The numbered step list appeared in both the banner and the step selection menu. Removed it from the banner so it only shows once in the menu.
- **— Removed duplicate label format listing from step5_consolidate.py**: Label file counts were shown twice — once as a summary and again in the selection menu. Removed the standalone summary so counts only appear in the menu.
- **— Fixed step4_labeling.py Qt plugin crash when run from yolo_env**: X-AnyLabeling failed with "Could not load the Qt platform plugin xcb" because `subprocess.Popen` inherited the full yolo_env environment. Shell-level unsets weren't enough. Now passes a clean `env` dict to Popen with only essential system variables (HOME, DISPLAY, etc.) and a PATH stripped of all venv entries, so the x-anylabeling_env activate script starts from a clean slate.
- **— Removed input directory requirement from step4_labeling.py**: X-AnyLabeling doesn't need the input directory passed to it. Removed the existence check and interactive prompt that blocked if the directory wasn't found. The directory path is still resolved and printed as a hint.
- **— Renamed all step files to remove stepX_ prefix**: `step1_extract_frames.py` → `extract_frames.py`, `step2_filter_images.py` → `filter_images.py`, `step3_anonymize.py` → `anonymize.py`, `step4_labeling.py` → `labeling.py`, `step5_consolidate.py` → `consolidate.py`, `step6_review_labels.py` → `review_labels.py`, `step_yolo_autolabel.py` → `yolo_autolabel.py`. Updated imports in `image_labeling_workflow.py`, `yolo_labeling_workflow.py`, and `CLAUDE.md`.
- **— Reordered yolo_labeling_workflow.py steps**: New order is: 1) Extract frames, 2) YOLO auto-label, 3) Anonymize, 4) Consolidate, 5) Review/correct labels (X-AnyLabeling), 6) Consolidate (final). Consolidation runs twice — before and after manual label review.
- **— Added existing output directory warning to consolidate.py**: Before proceeding, warns the user if the output directory already has files, showing the total file count and image count.
- **— Fixed consolidate output format for X-AnyLabeling compatibility in yolo workflow**: Step 4 (consolidate before review) now keeps images and labels together in the same folder (`separate_folders=False`) so X-AnyLabeling can find them. Step 6 (final consolidate) separates into Image/Label subfolders. Added `separate_folders` config key to `consolidate.py`. Also sets `labeling_input_dir` after step 4 so step 5 (X-AnyLabeling) shows the correct directory hint.

## 2026-01-30

- **17:10 — Updated workflow_config.json**: Added YOLO auto-labeling config values (yolo_model_path, autolabel_confidence, autolabel_iou, autolabel_imgsz, autolabel_device, autolabel_delete_unlabeled, autolabel_deleted_dir) with documentation, so both workflows share one config file.
- **17:00 — Created YOLO auto-labeling workflow**: Added `Pipeline/yolo_labeling_workflow.py` (orchestrator) and `Pipeline/step_yolo_autolabel.py` (YOLO inference step). This is a separate 5-step pipeline that uses a trained YOLO model to auto-label extracted frames, deletes images with no detections, anonymizes, opens X-AnyLabeling for review/correction, and consolidates the dataset. Reuses existing step1, step3, step4, and step5 modules.
- **16:18 — Fixed image_labeling_workflow.py**: Changed `from steps import ...` to direct module imports (`from step1_extract_frames import ...`, etc.) so the script works when run from the Pipeline directory. The previous import referenced a nonexistent `steps` package.
- **16:13 — Updated CLAUDE.md**: Added note that LOG.md writes are automatic and do not require user permission.
- **16:12 — Updated CLAUDE.md**: Updated logging policy to require time in addition to date for each log entry.
- **16:12 — Updated LOG.md**: Reformatted existing entries to include timestamps.
- **16:10 — Created CLAUDE.md**: Added project documentation covering structure, data flow, setup, pipeline usage, training, annotation formats, and key configuration.
- **16:10 — Created LOG.md**: Added this change log to track all modifications made to the project.
- **16:10 — Updated CLAUDE.md**: Added logging policy requiring all changes to be documented in LOG.md.
