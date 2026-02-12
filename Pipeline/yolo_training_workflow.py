#!/usr/bin/env python3
"""
YOLO Training Workflow

Alternative pipeline that uses a trained YOLO model to auto-label extracted
frames, deletes images without detections, then lets you review/correct
labels in X-AnyLabeling before consolidating the dataset and training a model.

Steps:
1. Extract image frames from video
2. YOLO auto-label (delete images with no detections)
3. Anonymize faces and license plates
4. Review/correct labels (X-AnyLabeling)
5. Consolidate dataset
6. Convert to YOLO format
7. Train YOLO model

Usage:
    python yolo_training_workflow.py
    python yolo_training_workflow.py --start-step 2
    python yolo_training_workflow.py --config yolo_workflow_config.json
    python yolo_training_workflow.py --video /path/to/video.mp4 --model /path/to/best.pt
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
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Import step modules
from extract_frames import extract_video_frames
from autolabel import yolo_autolabel
from anonymize import anonymize_images
from labeling import run_labeling
from consolidate import consolidate_files

# Import YOLO training utilities
try:
    from train_yolo import train_yolo
except ImportError as e:
    print(f"{RED}Warning: Could not import YOLO training module: {e}{RESET}")
    print(f"{RED}Make sure train_yolo.py is in the Pipeline directory{RESET}")
    train_yolo = None


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
    "autolabel_input_dir": "./extracted_frames",
    "autolabel_confidence": 0.25,
    "autolabel_iou": 0.45,
    "autolabel_imgsz": 640,
    "autolabel_device": "",
    "autolabel_delete_unlabeled": True,
    "autolabel_deleted_dir": "./deleted/unlabeled",
    "autolabel_output_dir": "./autolabeled",

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
    "anylabeling_repo": "~/Object_Detection/X-AnyLabeling",

    # Step 5 (final consolidation) -> YOLO conversion
    "yolo_train_ratio": 0.8,
    "yolo_classes_file": None,

    # Step 6: Train YOLO model
    "yolo_runs_dir": "/home/aidall/Object_Detection/runs/detect/runs",
    "train_epochs": 100,
    "train_batch": 16,
    "train_imgsz": 640,
    "train_device": "",
    "train_workers": 8,
    "train_project": "./runs",
    "train_name": None,
    "train_resume": False,
    "train_pretrained": True,
    "train_optimizer": "auto",
    "train_lr0": 0.01,
    "train_patience": 50,
    "train_cache": False,
    "train_amp": True,
    "train_augment": True,
}


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def print_banner():
    """Print the workflow banner."""
    print()
    print("=" * 60)
    print("          YOLO TRAINING WORKFLOW")
    print("=" * 60)
    print()


def print_step_menu():
    """Print the step selection menu."""
    print("Select the step to start from:")
    print()
    print("  1. Extract image frames from video")
    print("  2. YOLO auto-label (remove unlabeled images)")
    print("  3. Anonymize faces and license plates")
    print("  4. Review/correct labels (X-AnyLabeling)")
    print("  5. Consolidate & convert to YOLO format")
    print("  6. Train YOLO model")
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


def consolidate_and_convert(config, **kwargs):
    """Step 5: Consolidate with images and labels in separate folders, then convert to YOLO format."""
    config["separate_folders"] = True
    config["label_format"] = config.get("label_format", "json")
    config["skip_format_prompt"] = True
    config["convert_to_yolo"] = True

    # Remove stale YOLO output from previous runs
    consolidated_dir = config.get("consolidated_output_dir", "./Dataset")
    consolidated_path = Path(consolidated_dir)
    for stale_name in ("images", "labels", "classes.txt", "dataset.yaml"):
        stale_path = consolidated_path / stale_name
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.is_file():
            stale_path.unlink()

    print()
    print("=" * 60)
    print("CONSOLIDATING & CONVERTING TO YOLO FORMAT")
    print("=" * 60)

    result = consolidate_files(config, **kwargs)

    if result:
        # Store YAML path for summary
        yaml_path = consolidated_path / "dataset.yaml"
        if yaml_path.exists():
            config["yolo_dataset_yaml"] = str(yaml_path)

    return result


def train_yolo_model(config, **kwargs):
    """Step 6: Train YOLO model using the converted dataset."""
    if train_yolo is None:
        print(f"{RED}Error: YOLO training module not available.{RESET}")
        print(f"{RED}Make sure train_yolo.py is in the YOLO_Training directory.{RESET}")
        return False

    print()
    print("=" * 60)
    print("TRAINING YOLO MODEL")
    print("=" * 60)

    # Get dataset.yaml path
    yaml_path = config.get("yolo_dataset_yaml")

    # If Step 5 output exists, use it as default
    default_yaml = None
    if yaml_path and Path(yaml_path).exists():
        default_yaml = yaml_path
    else:
        # Try to find it in the Step 5 output directory
        consolidated_dir = config.get("consolidated_output_dir", "./Dataset")
        potential_yaml = Path(consolidated_dir) / "dataset.yaml"
        if potential_yaml.exists():
            default_yaml = str(potential_yaml)

    # Prompt user for dataset.yaml path
    print()
    if default_yaml:
        print(f"Default dataset.yaml from Step 5: {default_yaml}")
        user_input = input(f"{GREEN}Press Enter to use default, or enter path to dataset.yaml: {RESET}").strip()
        if user_input:
            yaml_path = user_input
        else:
            yaml_path = default_yaml
    else:
        print(f"{YELLOW}Warning: No dataset.yaml found from Step 5.{RESET}")
        user_input = input(f"{GREEN}Enter path to dataset.yaml: {RESET}").strip()
        if not user_input:
            print(f"{RED}Error: dataset.yaml path is required.{RESET}")
            return False
        yaml_path = user_input

    # Validate the path
    yaml_path_obj = Path(yaml_path)
    if not yaml_path_obj.exists():
        print(f"{RED}Error: dataset.yaml not found at {yaml_path}{RESET}")
        return False

    yaml_path = str(yaml_path_obj)
    print(f"\n{CYAN}Using dataset config: {yaml_path}{RESET}")
    print()

    # Determine version number and select base model
    from model_utils import find_yolo_versions, select_yolo_model
    runs_dir = Path(os.path.expanduser(config.get("yolo_runs_dir", "/home/aidall/Object_Detection/runs/detect/runs")))
    versions = find_yolo_versions(str(runs_dir))
    latest_version = max(versions.keys()) if versions else 0
    version_num = latest_version + 1
    train_name = f"YOLO_v{version_num}"

    # Select base model (select_yolo_model lists available versions)
    print(f"\n{CYAN}Select base model for training:{RESET}")
    train_model = select_yolo_model(runs_dir=str(runs_dir))
    if not train_model:
        train_model = "yolov8n.pt"
        print(f"{CYAN}Using default model: {train_model}{RESET}")

    print(f"\n{CYAN}New training run: {train_name}{RESET}")

    # Extract training parameters from config
    train_params = {
        "data": yaml_path,
        "model": train_model,
        "epochs": config.get("train_epochs", 100),
        "batch": config.get("train_batch", 16),
        "imgsz": config.get("train_imgsz", 640),
        "device": config.get("train_device", ""),
        "workers": config.get("train_workers", 8),
        "project": str(runs_dir),
        "name": train_name,
        "resume": config.get("train_resume", False),
        "pretrained": config.get("train_pretrained", True),
        "optimizer": config.get("train_optimizer", "auto"),
        "lr0": config.get("train_lr0", 0.01),
        "patience": config.get("train_patience", 50),
        "cache": config.get("train_cache", False),
        "amp": config.get("train_amp", True),
        "augment": config.get("train_augment", True),
    }

    try:
        results = train_yolo(**train_params)

        if results:
            # Store training results path
            config["training_results_dir"] = str(results.save_dir)
            print(f"\n{GREEN}Training completed successfully!{RESET}")
            print(f"Results saved to: {results.save_dir}")
            return True
        else:
            print(f"\n{RED}Training failed.{RESET}")
            return False

    except Exception as e:
        print(f"\n{RED}Error during training: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False


def print_progress(steps, current_step, start_step, status="running"):
    """Print a progress block showing all steps and current status."""
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CHECK = f"{GREEN}\u2714{RESET}"
    ARROW = f"{YELLOW}\u25b6{RESET}"
    SKIP = f"{DIM}\u2500{RESET}"
    PENDING = f"{DIM}\u25cb{RESET}"

    print(f"\n{CYAN}{'=' * 60}{RESET}")

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
        print(f"{CYAN}{'=' * 60}{RESET}")
        print(f"{BOLD}{GREEN}  WORKFLOW COMPLETE!{RESET}")
    else:
        print(f"{CYAN}{'=' * 60}{RESET}")


def _get_step_output_dir(step_num, config):
    """Return the output directory for a given step based on config."""
    mapping = {
        1: "extracted_frames_dir",
        2: "autolabel_output_result_dir",
        3: "anonymize_output_dir",
        4: "labeling_input_dir",
        5: "consolidated_output_dir",
        6: "training_results_dir",
    }
    key = mapping.get(step_num, "")
    return config.get(key, "")


def print_summary(results, config):
    """Print a final summary of all steps that were run."""
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
        (4, "Reviewing/correcting labels (X-AnyLabeling)", run_labeling),
        (5, "Consolidating & converting to YOLO format", consolidate_and_convert),
        (6, "Training YOLO model", train_yolo_model),
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

        result = step_func(config, from_previous_step=from_previous_step)

        # Normalize result: True/False/"skipped"/"stop"
        if result == "skipped":
            success = True
            skipped = True
        elif result == "stop":
            success = False
            skipped = False
        else:
            success = bool(result)
            skipped = False

        # Record result for summary
        results.append({
            "step": step_num,
            "name": step_name,
            "success": success,
            "output_dir": _get_step_output_dir(step_num, config),
        })

        if result == "stop":
            # User explicitly chose to stop
            print("\nWorkflow stopped.")
            print_summary(results, config)
            return False
        elif not success:
            print(f"\nStep {step_num} encountered an issue.")
            if step_num < 6:
                next_step_name = steps[step_num][1]
                continue_choice = input(f"{GREEN}Continue to Step {step_num + 1} - {next_step_name} anyway? (y/n): {RESET}").strip().lower()
                if continue_choice != 'y':
                    print("\nWorkflow stopped.")
                    print_summary(results, config)
                    return False
                from_previous_step = True
            else:
                print("\nWorkflow stopped.")
                print_summary(results, config)
                return False
        elif skipped:
            # Step was skipped, move directly to next step without prompting
            from_previous_step = True
        elif step_num < 6:
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
        description='YOLO Training Workflow - Extract frames, auto-label with YOLO, review, consolidate, and train model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactively
  python yolo_training_workflow.py

  # Start from a specific step
  python yolo_training_workflow.py --start-step 2

  # Use a specific configuration file
  python yolo_training_workflow.py --config yolo_workflow_config.json

  # Specify video and model
  python yolo_training_workflow.py --video /path/to/video.mp4 --model /path/to/best.pt

  # Generate a template configuration file
  python yolo_training_workflow.py --generate-config

Steps:
  1. Extract image frames from video using ffmpeg
  2. Run YOLO model to auto-label frames (delete images with no detections)
  3. Anonymize faces and license plates
  4. Review/correct labels in X-AnyLabeling
  5. Consolidate & convert to YOLO format
  6. Train YOLO model
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
