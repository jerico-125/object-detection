#!/usr/bin/env python3
"""
Image Labeling Workflow - A unified tool for the complete image labeling pipeline.

This program combines multiple tools into a single workflow:
1. Extract image frames from video (sample_video_frames_v4)
2. Filter out suitable images (interactive_image_viewer_v3)
3. Anonymize faces and license plates (understand-ai/anonymizer)
4. Label images (X-anylabeling)
5. Consolidate files
6. Review labels

Usage:
    python image_labeling_workflow.py
    python image_labeling_workflow.py --start-step 3
    python image_labeling_workflow.py --config workflow_config.json

"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Import step modules
from extract_frames import extract_video_frames
from filter_images import filter_images
from anonymize import anonymize_images
from labeling import run_labeling
from consolidate import consolidate_files
from review_labels import review_labels


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
    "blur_threshold": 100.0,  # Frames with Laplacian variance below this are considered blurry
    "clustering_eps": None,  # DBSCAN eps (max neighbor distance); None = auto from 1.0 - frame_threshold
    "clustering_min_samples": 2,  # DBSCAN min_samples parameter
    "clustering_deleted_dir": "./deleted/clustering",  # Directory for non-representative/blurry images

    # Step 2: Image filtering
    "filter_input_dir": "./extracted_frames",
    "kept_images_dir": "./kept_images",
    "deleted_images_dir": "./deleted/filtered",
    "move_to_trash": True,
    "image_extensions": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],

    # Step 3: Anonymization (faces and license plates)
    "anonymize_input_dir": "./kept_images",
    "anonymize_output_dir": "./anonymized_images",
    "anonymizer_weights_dir": "./anonymizer_weights",
    "face_threshold": 0.3,  # Detection threshold for faces (0.0-1.0, lower = more sensitive)
    "plate_threshold": 0.3,  # Detection threshold for license plates (0.0-1.0)
    "obfuscation_kernel": "65,3,19",  # kernel_size, sigma, box_kernel_size

    # Step 4: Labeling (X-anylabeling)
    "labeling_input_dir": "./anonymized_images",
    "anylabeling_venv": "x-anylabeling_env",  # Name of the venv containing X-anylabeling
    "anylabeling_repo": "~/AI_Hub/X-AnyLabeling",  # Path to the X-AnyLabeling source repo
    "deleted_empty_labels_dir": "./deleted/empty_labels",  # Directory for images with empty/missing labels

    # Step 5: Consolidate files and review labels
    "consolidated_output_dir": "./Dataset",
    "include_labels": True,
    "copy_files": True,
    "review_output_dir": "./reviewed_images",
}


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def print_banner():
    """Print the workflow banner."""
    print()
    print("=" * 70)
    print("            IMAGE LABELING WORKFLOW")
    print("=" * 70)
    print()
    print("  This tool combines the complete image labeling pipeline:")
    print()
    print("    1. Extract image frames from video")
    print("    2. Filter out suitable images")
    print("    3. Anonymize faces and license plates")
    print("    4. Label images (X-anylabeling)")
    print("    5. Consolidate files")
    print("    6. Review labels")
    print()
    print("=" * 70)
    print()


def print_step_menu():
    """Print the step selection menu."""
    print("Select the step to start from:")
    print()
    print("  1. Extracting image frames from video")
    print("  2. Filtering out suitable images")
    print("  3. Anonymizing faces and license plates")
    print("  4. Labeling")
    print("  5. Consolidate files")
    print("  6. Review labels")
    print()
    print("  0. Exit")
    print()


def get_step_choice() -> int:
    """Get the user's step choice."""
    while True:
        try:
            choice = input(f"{GREEN}Enter step number (1-6, or 0 to exit): {RESET}").strip()
            if choice == "":
                return 1  # Default to step 1
            choice = int(choice)
            if 0 <= choice <= 6:
                return choice
            print("Please enter a number between 0 and 6.")
        except ValueError:
            print("Please enter a valid number.")


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
        2: "kept_images_dir",
        3: "anonymize_output_dir",
        4: "labeling_input_dir",
        5: "consolidated_output_dir",
        6: "review_output_dir",
    }
    key = mapping.get(step_num, "")
    return config.get(key, "")


def print_summary(results):
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

    print(f"{CYAN}{'=' * 60}{RESET}")


def run_workflow(start_step: int, config: Dict[str, Any]) -> bool:
    """Run the workflow starting from the specified step."""
    steps = [
        (1, "Extracting image frames from video", extract_video_frames),
        (2, "Filtering out suitable images", filter_images),
        (3, "Anonymizing faces and license plates", anonymize_images),
        (4, "Labeling", run_labeling),
        (5, "Consolidate files", consolidate_files),
        (6, "Review labels", review_labels),
    ]

    # Track if we're continuing from a previous step
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

        # Pass from_previous_step to indicate if input path is known
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
                print_summary(results)
                return False

        # Prompt to continue after each step
        if step_num < 6:
            next_step_name = steps[step_num][1]  # Get next step's name (index is step_num since list is 0-indexed)
            print()
            next_choice = input(f"{GREEN}Step {step_num} done. Continue to Step {step_num + 1} - {next_step_name}? (y/n): {RESET}").strip().lower()
            if next_choice != 'y':
                print_progress(steps, step_num, start_step, "done")
                print_summary(results)
                return True
            # If continuing, mark that we're coming from previous step
            from_previous_step = True

    # All steps complete
    print_progress(steps, 6, start_step, "complete")
    print_summary(results)
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
        Path("workflow_config.json"),  # Current directory
        script_dir / "workflow_config.json",  # Script directory
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
        description='Image Labeling Workflow - Complete pipeline for video to labeled images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactively (will prompt for step selection)
  python image_labeling_workflow.py

  # Start from a specific step
  python image_labeling_workflow.py --start-step 3

  # Use a specific configuration file
  python image_labeling_workflow.py --config my_config.json

  # Generate a template configuration file
  python image_labeling_workflow.py --generate-config

Steps:
  1. Extract image frames from video using ffmpeg
  2. Interactively filter/select suitable images
  3. Anonymize faces and license plates (understand-ai/anonymizer)
  4. Label images using X-anylabeling
  5. Consolidate files
  6. Review labeled images
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
        config_path = "workflow_config.json"
        save_config(DEFAULT_CONFIG, config_path)
        print("Template configuration file generated. Edit it to customize your workflow.")
        return

    # Print banner
    print_banner()

    # Determine config file to use
    if args.config:
        # User specified a config file
        config_path = args.config
        print(f"Using config file: {config_path}")
    else:
        # Look for default config file
        default_config_path = find_default_config()
        if default_config_path:
            config_path = prompt_with_default("Enter config file path or press Enter to use default", default_config_path)
        else:
            print("No default config file found (workflow_config.json)")
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

    # Override video path if provided
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
