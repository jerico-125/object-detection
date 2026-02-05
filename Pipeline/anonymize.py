#!/usr/bin/env python3
"""
Step 3: Anonymize Faces and License Plates

Uses the understand-ai/anonymizer library to blur faces and license plates in images.
Falls back to CLI instructions if the library is not installed.
"""

import os
from pathlib import Path
from typing import Dict, Any

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# ============================================================================
# USER INTERACTION UTILITIES
# ============================================================================

def list_subdirectories(base_dir: str) -> list:
    """List subdirectories in a base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    return [d.name for d in base_path.iterdir() if d.is_dir()]


def prompt_with_directory_options(prompt_text: str, base_dir: str, default_subdir: str = "") -> str:
    """
    Prompt user for a directory, showing available subdirectories if they exist.
    Returns the full path to the selected directory.
    """
    subdirs = list_subdirectories(base_dir)

    if subdirs:
        print(f"\nAvailable folders in {base_dir}:")
        for i, subdir in enumerate(subdirs, 1):
            print(f"  {i}. {subdir}")
        print()

        # Determine default
        if default_subdir and default_subdir in subdirs:
            default_path = f"{base_dir}/{default_subdir}"
        elif len(subdirs) == 1:
            default_path = f"{base_dir}/{subdirs[0]}"
        else:
            default_path = base_dir

        user_input = input(f"{GREEN}{prompt_text} [default: {default_path}]: {RESET}").strip()

        if not user_input:
            return default_path

        # Check if user entered a number to select from list
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(subdirs):
                return f"{base_dir}/{subdirs[idx]}"

        return user_input
    else:
        # No subdirectories found, just prompt with base_dir as default
        user_input = input(f"{GREEN}{prompt_text} [default: {base_dir}]: {RESET}").strip()
        return user_input if user_input else base_dir


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def anonymize_images(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Step 3: Anonymize faces and license plates in images."""

    # Get video name from previous steps
    video_name = config.get("video_name", "")
    if not video_name and config.get("video_path"):
        video_name = Path(config.get("video_path")).stem

    # Determine input directory
    if from_previous_step:
        if config.get("labeling_input_dir") and os.path.exists(config["labeling_input_dir"]):
            input_dir = config["labeling_input_dir"]
        else:
            base_input = config.get("anonymize_input_dir", "./extracted_frames")
            if video_name:
                input_dir = f"{base_input}/{video_name}"
            else:
                input_dir = base_input
    else:
        base_dir = config.get("anonymize_input_dir", "./extracted_frames")
        input_dir = prompt_with_directory_options(
            "Enter the directory containing images to anonymize",
            base_dir,
            video_name
        )
        if not input_dir:
            print(f"{RED}No directory provided.{RESET}")
            return False
        # Update video_name if not set
        if not video_name:
            video_name = Path(input_dir).name
            config["video_name"] = video_name

    if not os.path.exists(input_dir):
        print(f"{RED}Error: Input directory '{input_dir}' does not exist.{RESET}")
        return False

    # Set up output directory
    base_output = config.get("anonymize_output_dir", "./anonymized_images")
    if video_name:
        output_dir = f"{base_output}/{video_name}"
    else:
        output_dir = base_output

    weights_dir = config.get("anonymizer_weights_dir", "./anonymizer_weights")
    face_threshold = config.get("face_threshold", 0.3)
    plate_threshold = config.get("plate_threshold", 0.3)
    obfuscation_kernel = config.get("obfuscation_kernel", "65,3,19")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Weights directory: {weights_dir}")
    print(f"Face detection threshold: {face_threshold}")
    print(f"Plate detection threshold: {plate_threshold}")
    print(f"Obfuscation kernel: {obfuscation_kernel}")
    print()

    # Check if anonymizer is available
    try:
        from anonymizer.detection import Detector
        from anonymizer.obfuscation import Obfuscator
        from anonymizer.anonymization import Anonymizer
        anonymizer_available = True
    except ImportError:
        anonymizer_available = False

    if anonymizer_available:
        print("Anonymizer library found. Running anonymization...")
        print()

        try:
            # Parse obfuscation kernel parameters
            kernel_parts = [int(x.strip()) for x in obfuscation_kernel.split(",")]
            kernel_size = kernel_parts[0] if len(kernel_parts) > 0 else 65
            sigma = kernel_parts[1] if len(kernel_parts) > 1 else 3
            box_kernel_size = kernel_parts[2] if len(kernel_parts) > 2 else 19

            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(weights_dir).mkdir(parents=True, exist_ok=True)

            # Initialize obfuscator
            obfuscator = Obfuscator(
                kernel_size=kernel_size,
                sigma=sigma,
                box_kernel_size=box_kernel_size
            )

            # Initialize detectors
            detectors = {
                'face': Detector(kind='face', weights_path=weights_dir),
                'plate': Detector(kind='plate', weights_path=weights_dir)
            }

            # Create anonymizer
            anonymizer = Anonymizer(obfuscator=obfuscator, detectors=detectors)

            # Run anonymization
            detection_thresholds = {
                'face': face_threshold,
                'plate': plate_threshold
            }

            extensions = config.get("image_extensions", ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])

            anonymizer.anonymize_images(
                input_path=input_dir,
                output_path=output_dir,
                detection_thresholds=detection_thresholds,
                file_types=extensions,
                write_json=False
            )

            print()
            print("-" * 50)
            print("Anonymization complete!")
            print(f"Output directory: {output_dir}")

            # Update config for next step
            config["labeling_input_dir"] = output_dir

            return True

        except Exception as e:
            print(f"{RED}Error during anonymization: {e}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    else:
        # Anonymizer not installed - offer CLI alternative
        print("=" * 60)
        print("ANONYMIZER LIBRARY NOT FOUND")
        print("=" * 60)
        print()
        print("The understand-ai/anonymizer library is not installed.")
        print()
        print("To install it:")
        print("  1. Clone the repository:")
        print("     git clone https://github.com/understand-ai/anonymizer")
        print("  2. Install dependencies:")
        print("     cd anonymizer && pip install -r requirements.txt")
        print()
        print("Or run manually with CLI:")
        print(f"  PYTHONPATH=$PYTHONPATH:. python anonymizer/bin/anonymize.py \\")
        print(f"    --input {input_dir} \\")
        print(f"    --image-output {output_dir} \\")
        print(f"    --weights {weights_dir} \\")
        print(f"    --face-threshold={face_threshold} \\")
        print(f"    --plate-threshold={plate_threshold} \\")
        print(f"    --obfuscation-kernel=\"{obfuscation_kernel}\" \\")
        print(f"    --no-write-detections")
        print()

        skip_choice = input(f"{GREEN}Skip anonymization and continue? (y/n): {RESET}").strip().lower()
        if skip_choice == 'y':
            # Use input_dir as output for next step
            config["labeling_input_dir"] = input_dir
            return True
        else:
            print("\nPlease install anonymizer and try again.")
            return False
