#!/usr/bin/env python3
"""
Step 4: Image Labeling with X-AnyLabeling

Launches the X-AnyLabeling tool for interactive image annotation.
Searches for the virtual environment and runs the labeling application.
"""

import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
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
# EMPTY LABEL CLEANUP
# ============================================================================

def _is_label_empty(label_path: Path) -> bool:
    """Check if a label file has no annotations."""
    suffix = label_path.suffix.lower()

    if suffix == ".json":
        try:
            with open(label_path, "r") as f:
                data = json.load(f)
            return len(data.get("shapes", [])) == 0
        except (json.JSONDecodeError, KeyError):
            return True

    elif suffix == ".txt":
        try:
            return label_path.read_text().strip() == ""
        except Exception:
            return True

    elif suffix == ".xml":
        try:
            tree = ET.parse(label_path)
            root = tree.getroot()
            return len(root.findall("object")) == 0
        except ET.ParseError:
            return True

    return False


def remove_empty_labels(input_dir: str, label_format: str = "json",
                        image_extensions: list = None,
                        deleted_dir: str = "./deleted/empty_labels") -> int:
    """
    Remove images and their label files when the label has no annotations.

    Moves affected files to the deleted directory instead of permanently deleting.

    Args:
        input_dir: Directory containing images and label files.
        label_format: Label format — 'json', 'txt', or 'xml'.
        image_extensions: List of image extensions to consider.
        deleted_dir: Directory to move empty-label images/labels to.

    Returns:
        Number of image/label pairs removed.
    """
    if image_extensions is None:
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

    ext_map = {"json": ".json", "txt": ".txt", "xml": ".xml"}
    label_ext = ext_map.get(label_format, ".json")

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"{RED}Directory not found: {input_dir}{RESET}")
        return 0

    deleted_path = Path(deleted_dir)
    deleted_path.mkdir(parents=True, exist_ok=True)

    removed = 0

    # Gather all image files (recursive search through subdirectories)
    image_files = [
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lstrip(".").lower() in image_extensions
    ]

    for img_file in sorted(image_files):
        label_file = img_file.with_suffix(label_ext)

        # If no label file exists, the image was never labeled — treat as empty
        if not label_file.exists() or _is_label_empty(label_file):
            # Preserve subdirectory structure in deleted directory
            rel_path = img_file.relative_to(input_path)
            dest_dir = deleted_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_file), str(dest_dir / img_file.name))
            if label_file.exists():
                shutil.move(str(label_file), str(dest_dir / label_file.name))
            removed += 1

    return removed


def _prompt_remove_empty(default_dir: str, config: Dict[str, Any]) -> None:
    """Prompt the user to remove images with empty/missing labels."""
    print()
    response = input(
        f"{GREEN}Remove images with empty or missing labels? (y/n) [default: n]: {RESET}"
    ).strip().lower()

    if response not in ("y", "yes"):
        print("Skipping empty label cleanup.")
        return

    # Ask user which directory to clean up
    target_dir = input(
        f"{GREEN}Enter directory to check for empty labels [default: {default_dir}]: {RESET}"
    ).strip()
    if not target_dir:
        target_dir = default_dir

    if not os.path.exists(target_dir):
        print(f"{RED}Directory not found: {target_dir}{RESET}")
        return

    label_format = config.get("label_format", "json")
    image_extensions = config.get("image_extensions",
                                  ["jpg", "jpeg", "png", "bmp", "tiff", "webp"])
    deleted_dir = config.get("deleted_empty_labels_dir", "./deleted/empty_labels")
    removed = remove_empty_labels(target_dir, label_format, image_extensions, deleted_dir)
    if removed > 0:
        print(f"Moved {removed} unlabeled image(s) and their label files to: {deleted_dir}")
    else:
        print("No empty labels found. All images have annotations.")


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def run_labeling(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Step 4: Run X-anylabeling for image labeling."""

    # Get video name (from previous steps or derived from video_path)
    video_name = config.get("video_name", "")
    if not video_name and config.get("video_path"):
        video_name = Path(config.get("video_path")).stem

    # Determine input directory - prefer labeling_input_dir if explicitly set
    # When set by a previous step (e.g. consolidate), it's already the final path
    if from_previous_step and config.get("labeling_input_dir") and os.path.exists(config["labeling_input_dir"]):
        input_dir = config["labeling_input_dir"]
    elif config.get("anonymize_output_dir") and os.path.exists(config.get("anonymize_output_dir")):
        base_dir = config.get("anonymize_output_dir", "./anonymized_images")
        input_dir = f"{base_dir}/{video_name}" if video_name else base_dir
    else:
        base_dir = config.get("labeling_input_dir", "./anonymized_images")
        input_dir = f"{base_dir}/{video_name}" if video_name else base_dir

    venv_name = config.get("anylabeling_venv", "x-anylabeling_env")

    # Convert to absolute path for display
    input_dir = os.path.abspath(input_dir)

    # Find the venv activation script
    home_dir = os.path.expanduser("~")
    possible_venv_paths = [
        os.path.join(home_dir, venv_name),
        os.path.join(home_dir, "venvs", venv_name),
        os.path.join(home_dir, ".venvs", venv_name),
        os.path.join(os.getcwd(), venv_name),
        venv_name,  # Could be absolute path
    ]

    venv_path = None
    for path in possible_venv_paths:
        activate_script = os.path.join(path, "bin", "activate")
        if os.path.exists(activate_script):
            venv_path = path
            break

    if venv_path:
        print(f"Found venv: {venv_path}")
        print("Launching X-anylabeling...")
        print()
        print("=" * 60)
        print("X-ANYLABELING INSTRUCTIONS")
        print("=" * 60)
        print()
        print(f"1. Open/load images from: {input_dir}")
        print("2. Label your images")
        print("3. Save labels (they will be saved as JSON files)")
        print("4. Close X-anylabeling when done")
        print()
        print("=" * 60)
        print()

        # Build the command to activate venv and run anylabeling
        activate_script = os.path.join(venv_path, "bin", "activate")
        repo_path = os.path.expanduser(config.get("anylabeling_repo", "~/AI_Hub/X-AnyLabeling"))
        cmd = (
            f'source "{activate_script}" && '
            f'PYTHONPATH="{repo_path}:$PYTHONPATH" python -m anylabeling.app'
        )

        # Build a clean environment to avoid yolo_env Qt/OpenCV conflicts
        clean_env = {}
        # Keep essential system variables
        for key in ('HOME', 'USER', 'LANG', 'TERM', 'DISPLAY', 'WAYLAND_DISPLAY',
                     'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS', 'SHELL',
                     'SSH_AUTH_SOCK', 'SSH_CONNECTION', 'SSH_TTY'):
            if key in os.environ:
                clean_env[key] = os.environ[key]
        # Build PATH without any venv entries
        system_path = ':'.join(
            p for p in os.environ.get('PATH', '/usr/bin:/bin').split(':')
            if 'venv' not in p and 'env/' not in p
        )
        clean_env['PATH'] = system_path

        try:
            # Run in a bash shell to handle source command
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                cwd=os.path.expanduser("~"),
                env=clean_env
            )
            print(f"X-anylabeling started (PID: {process.pid})")
            print("Waiting for X-anylabeling to close...")
            process.wait()
            print("\nX-anylabeling closed.")

            # Offer to remove unlabeled images
            _prompt_remove_empty(input_dir, config)
            return True
        except Exception as e:
            print(f"{RED}Error running X-anylabeling: {e}{RESET}")
            return False
    else:
        print("=" * 60)
        print("X-ANYLABELING VENV NOT FOUND")
        print("=" * 60)
        print()
        print(f"{RED}Could not find virtual environment: {venv_name}{RESET}")
        print()
        print("Searched in:")
        for path in possible_venv_paths:
            print(f"  - {path}")
        print()
        print("Please run X-anylabeling manually:")
        print(f"  1. Activate your venv: source ~/{venv_name}/bin/activate")
        print("  2. Run: python -m anylabeling.app")
        print(f"  3. Open images from: {input_dir}")
        print()
        input(f"{GREEN}Press Enter when labeling is complete to continue...{RESET}")

        # Offer to remove unlabeled images
        _prompt_remove_empty(input_dir, config)
        return True
