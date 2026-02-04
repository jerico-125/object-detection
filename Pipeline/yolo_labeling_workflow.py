#!/usr/bin/env python3
"""
YOLO Auto-Labeling Workflow

Alternative pipeline that uses a trained YOLO model to auto-label extracted
frames, deletes images without detections, then lets you review/correct
labels in X-AnyLabeling before consolidating the dataset.

Steps:
1. Extract image frames from video
2. YOLO auto-label (delete images with no detections)
3. Anonymize faces and license plates
4. Consolidate files
5. Review/correct labels (X-AnyLabeling)
6. Consolidate files (final)

Usage:
    python yolo_labeling_workflow.py
    python yolo_labeling_workflow.py --start-step 2
    python yolo_labeling_workflow.py --config yolo_workflow_config.json
    python yolo_labeling_workflow.py --video /path/to/video.mp4 --model /path/to/best.pt
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Import step modules
from extract_frames import extract_video_frames
from yolo_autolabel import yolo_autolabel
from anonymize import anonymize_images
from labeling import run_labeling
from consolidate import consolidate_files

# Add YOLO_Training to path for convert_json_to_yolo import
_yolo_training_dir = str(Path(__file__).parent.parent / "YOLO_Training")
if _yolo_training_dir not in sys.path:
    sys.path.insert(0, _yolo_training_dir)
from convert_json_to_yolo import convert_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Step 1: Video frame extraction
    "video_path": "",
    "extracted_frames_dir": "./extracted_frames",
    "frame_threshold": 0.85,
    "target_fps": 3.0,
    "frame_interval": None,
    "max_frames": None,
    "histogram_bins": 32,
    "frame_prefix": "frame",
    "blur_threshold": 100.0,
    "clustering": False,
    "clustering_eps": None,
    "clustering_min_samples": 2,

    # Step 2: YOLO auto-labeling
    "yolo_model_path": "",
    "autolabel_input_dir": "./extracted_frames",
    "autolabel_confidence": 0.25,
    "autolabel_iou": 0.45,
    "autolabel_imgsz": 640,
    "autolabel_device": "",
    "autolabel_delete_unlabeled": True,
    "autolabel_deleted_dir": "./deleted/unlabeled",

    # Step 3: Anonymization
    "anonymize_input_dir": "./extracted_frames",
    "anonymize_output_dir": "./anonymized_images",
    "anonymizer_weights_dir": "./anonymizer_weights",
    "face_threshold": 0.3,
    "plate_threshold": 0.3,
    "obfuscation_kernel": "65,3,19",

    # Step 4: Consolidate files
    "consolidated_output_dir": "./Dataset",
    "include_labels": True,
    "copy_files": True,
    "label_format": "json",
    "image_extensions": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],

    # Step 5: Review/correct labels in X-AnyLabeling
    "labeling_input_dir": "./anonymized_images",
    "anylabeling_venv": "x-anylabeling_env",
    "anylabeling_repo": "~/AI_Hub/X-AnyLabeling",

    # Step 6 (final consolidation) -> YOLO conversion
    "yolo_train_ratio": 0.8,
    "yolo_classes_file": None,
}


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def print_banner():
    """Print the workflow banner."""
    print()
    print("=" * 70)
    print("            YOLO AUTO-LABELING WORKFLOW")
    print("=" * 70)
    print()


def print_step_menu():
    """Print the step selection menu."""
    print("Select the step to start from:")
    print()
    print("  1. Extract image frames from video")
    print("  2. YOLO auto-label (remove unlabeled images)")
    print("  3. Anonymize faces and license plates")
    print("  4. Consolidate files")
    print("  5. Review/correct labels (X-AnyLabeling)")
    print("  6. Consolidate files (final)")
    print()
    print("  0. Exit")
    print()


def get_step_choice() -> int:
    """Get the user's step choice."""
    while True:
        try:
            choice = input(f"{GREEN}Enter step number (1-6, or 0 to exit): {RESET}").strip()
            if choice == "":
                return 1
            choice = int(choice)
            if 0 <= choice <= 6:
                return choice
            print("Please enter a number between 0 and 6.")
        except ValueError:
            print("Please enter a valid number.")


def consolidate_together(config, **kwargs):
    """Step 4: Consolidate with images and labels in the same folder."""
    config["separate_folders"] = False
    result = consolidate_files(config, **kwargs)
    # Set labeling_input_dir so X-AnyLabeling step knows where to find files
    if result and config.get("consolidated_output_dir"):
        config["labeling_input_dir"] = config["consolidated_output_dir"]
    return result


def consolidate_separated(config, **kwargs):
    """Step 6: Consolidate with images and labels in separate folders, then convert to YOLO format."""
    config["separate_folders"] = True
    config["label_format"] = config.get("label_format", "json")
    config["skip_format_prompt"] = True
    config["convert_to_yolo"] = True
    result = consolidate_files(config, **kwargs)

    if not result:
        return False

    # Convert consolidated JSON labels to YOLO txt format in the same output directory
    # consolidate_files stores the actual output dir chosen by the user in review_input_dir
    consolidated_dir = config.get("review_input_dir", config.get("consolidated_output_dir", "./Dataset"))
    train_ratio = config.get("yolo_train_ratio", 0.8)
    classes_file = config.get("yolo_classes_file")
    # Remove stale YOLO output from previous runs so convert_dataset starts fresh
    consolidated_path = Path(consolidated_dir)
    for stale_name in ("images", "labels", "classes.txt", "dataset.yaml"):
        stale_path = consolidated_path / stale_name
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.is_file() and not (stale_name == "classes.txt" and classes_file):
            stale_path.unlink()

    print()
    print("#" * 70)
    print("# CONVERTING JSON LABELS TO YOLO FORMAT")
    print("#" * 70)
    print(f"\nInput/output directory: {consolidated_dir}")
    print(f"Train/val split:       {train_ratio}/{round(1 - train_ratio, 2)}")

    try:
        yaml_path = convert_dataset(
            input_dir=consolidated_dir,
            output_dir=consolidated_dir,
            train_ratio=train_ratio,
            classes_file=classes_file,
        )
        if yaml_path:
            # Remove intermediate Image/ and Label/ folders — YOLO images/ and labels/ are the final output
            consolidated_path = Path(consolidated_dir)
            for folder_name in ("Image", "Label"):
                folder = consolidated_path / folder_name
                if folder.exists() and folder.is_dir():
                    shutil.rmtree(folder)
                    print(f"Removed intermediate folder: {folder}")

            print(f"\nYOLO dataset ready for training.")
            print(f"Use with: python train_yolo.py --data {yaml_path}")
            config["yolo_dataset_yaml"] = yaml_path
    except Exception as e:
        print(f"{RED}Error during YOLO conversion: {e}{RESET}")
        return False

    return True


def print_progress(steps, current_step, start_step, status="running"):
    """Print a progress block showing all steps and current status."""
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CHECK = f"{GREEN}\u2714{RESET}"
    ARROW = f"{YELLOW}\u25b6{RESET}"
    SKIP = f"{DIM}\u2500{RESET}"
    PENDING = f"{DIM}\u25cb{RESET}"

    print(f"\n{CYAN}{'=' * 50}{RESET}")

    for step_num, step_name, _ in steps:
        if step_num < start_step:
            # Skipped steps (user jumped past these)
            icon = SKIP
            style = DIM
        elif step_num < current_step:
            # Completed steps (actually ran)
            icon = CHECK
            style = ""
        elif step_num == current_step and status == "running":
            icon = ARROW
            style = BOLD
        elif step_num == current_step and status in ("done", "complete"):
            icon = CHECK
            style = ""
        else:
            icon = PENDING
            style = DIM

        print(f"  {icon} {style}Step {step_num}: {step_name}{RESET if style else ''}")

    if status == "complete":
        print(f"{CYAN}{'=' * 50}{RESET}")
        print(f"{BOLD}{GREEN}  WORKFLOW COMPLETE!{RESET}")
    else:
        print(f"{CYAN}{'=' * 50}{RESET}")


def _get_step_output_dir(step_num, config):
    """Return the output directory for a given step based on config."""
    mapping = {
        1: "extracted_frames_dir",
        2: "autolabel_input_dir",
        3: "anonymize_output_dir",
        4: "consolidated_output_dir",
        5: "labeling_input_dir",
        6: "consolidated_output_dir",
    }
    key = mapping.get(step_num, "")
    return config.get(key, "")


def print_summary(results, config):
    """Print a final summary of all steps that were run."""
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    print()
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}  WORKFLOW SUMMARY{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")

    for r in results:
        status_icon = f"{GREEN}\u2714{RESET}" if r["success"] else f"{RED}\u2718{RESET}"
        print(f"  {status_icon} Step {r['step']}: {r['name']}")
        if r["output_dir"]:
            print(f"      \u2192 {r['output_dir']}")

    # Show YOLO dataset path if conversion was done
    yaml_path = config.get("yolo_dataset_yaml")
    if yaml_path:
        print(f"\n  YOLO dataset: {yaml_path}")

    print(f"{CYAN}{'=' * 60}{RESET}")


def run_workflow(start_step: int, config: Dict[str, Any]) -> bool:
    """Run the workflow starting from the specified step."""
    steps = [
        (1, "Extracting image frames from video", extract_video_frames),
        (2, "YOLO auto-labeling", yolo_autolabel),
        (3, "Anonymizing faces and license plates", anonymize_images),
        (4, "Consolidating files", consolidate_together),
        (5, "Reviewing/correcting labels (X-AnyLabeling)", run_labeling),
        (6, "Consolidating files (final)", consolidate_separated),
    ]

    from_previous_step = False
    results = []

    # Print initial progress display
    print()
    print_progress(steps, start_step, start_step, "running")

    for step_num, step_name, step_func in steps:
        if step_num < start_step:
            continue

        # Update progress display to show current step
        if step_num > start_step:
            print_progress(steps, step_num, start_step, "running")

        success = step_func(config, from_previous_step=from_previous_step)

        # Record result for summary
        results.append({
            "step": step_num,
            "name": step_name,
            "success": success,
            "output_dir": _get_step_output_dir(step_num, config),
        })

        if not success:
            print(f"\nStep {step_num} encountered an issue.")
            continue_choice = input(f"{GREEN}Continue to next step anyway? (y/n): {RESET}").strip().lower()
            if continue_choice != 'y':
                print("\nWorkflow stopped.")
                print_summary(results, config)
                return False

        if step_num < 6:
            next_step_name = steps[step_num][1]
            print()
            next_choice = input(f"{GREEN}Step {step_num} done. Continue to Step {step_num + 1} - {next_step_name}? (y/n): {RESET}").strip().lower()
            if next_choice != 'y':
                print_progress(steps, step_num, start_step, "done")
                print_summary(results, config)
                return True
            from_previous_step = True

    # All steps complete
    print_progress(steps, 6, start_step, "complete")
    print_summary(results, config)
    return True


# ============================================================================
# CONFIG UTILITIES
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    except FileNotFoundError:
        print(f"{RED}Config file not found: {config_path}{RESET}")
        return DEFAULT_CONFIG.copy()
    except json.JSONDecodeError as e:
        print(f"{RED}Error parsing config file: {e}{RESET}")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def prompt_with_default(prompt_text: str, default_value: str) -> str:
    """Prompt user for input, showing default value. Returns default if empty input."""
    if default_value:
        user_input = input(f"{GREEN}{prompt_text} [default: {default_value}]: {RESET}").strip()
    else:
        user_input = input(f"{GREEN}{prompt_text}: {RESET}").strip()
    return user_input if user_input else default_value


def find_default_config() -> Optional[str]:
    """Find the default config file in common locations."""
    script_dir = Path(__file__).parent
    possible_configs = [
        Path("yolo_workflow_config.json"),
        script_dir / "yolo_workflow_config.json",
        Path("workflow_config.json"),
        script_dir / "workflow_config.json",
    ]

    for config_path in possible_configs:
        if config_path.exists():
            return str(config_path)
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='YOLO Auto-Labeling Workflow - Extract frames, auto-label with YOLO, review, consolidate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactively
  python yolo_labeling_workflow.py

  # Start from a specific step
  python yolo_labeling_workflow.py --start-step 2

  # Use a specific configuration file
  python yolo_labeling_workflow.py --config yolo_workflow_config.json

  # Specify video and model
  python yolo_labeling_workflow.py --video /path/to/video.mp4 --model /path/to/best.pt

  # Generate a template configuration file
  python yolo_labeling_workflow.py --generate-config

Steps:
  1. Extract image frames from video using ffmpeg
  2. Run YOLO model to auto-label frames (delete images with no detections)
  3. Anonymize faces and license plates
  4. Consolidate files into dataset
  5. Review/correct labels in X-AnyLabeling
  6. Consolidate files (final)
        """
    )

    parser.add_argument('--start-step', '-s', type=int, choices=[1, 2, 3, 4, 5, 6],
                       help='Start from this step (1-6)')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration JSON file')
    parser.add_argument('--generate-config', '-g', action='store_true',
                       help='Generate a template configuration file')
    parser.add_argument('--video', '-v', type=str,
                       help='Path to input video file (for step 1)')
    parser.add_argument('--model', '-m', type=str,
                       help='Path to YOLO model weights (for step 2)')

    args = parser.parse_args()

    # Generate config template
    if args.generate_config:
        config_path = "yolo_workflow_config.json"
        save_config(DEFAULT_CONFIG, config_path)
        print("Template configuration file generated. Edit it to customize your workflow.")
        return

    # Print banner
    print_banner()

    # Determine config file to use
    if args.config:
        config_path = args.config
        print(f"Using config file: {config_path}")
    else:
        default_config_path = find_default_config()
        if default_config_path:
            config_path = prompt_with_default("Enter config file path or press Enter to use default", default_config_path)
        else:
            print("No default config file found (yolo_workflow_config.json)")
            config_path = input(f"{GREEN}Enter config file path (or press Enter to use built-in defaults): {RESET}").strip()

    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        if config_path:
            print(f"{RED}Config file not found: {config_path}{RESET}")
        print("Using built-in default configuration.")
        config = DEFAULT_CONFIG.copy()

    # Override from command-line args
    if args.video:
        config["video_path"] = args.video
    if args.model:
        config["yolo_model_path"] = args.model

    print()

    # Determine starting step
    if args.start_step:
        start_step = args.start_step
        print(f"Starting from Step {start_step}")
    else:
        print_step_menu()
        start_step = get_step_choice()
        if start_step == 0:
            print("Exiting.")
            return

    # Run the workflow
    try:
        run_workflow(start_step, config)
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
