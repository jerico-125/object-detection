"""
Shared YOLO model discovery and interactive selection utilities.

Used by autolabel.py, inference.py, and main.py.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Default directory for YOLO version runs
DEFAULT_RUNS_DIR = "/home/aidall/Object_Detection/runs/detect/runs"


def find_yolo_versions(runs_dir: str = DEFAULT_RUNS_DIR) -> Dict[int, str]:
    """Find all YOLO_v* models in the runs directory.

    Returns:
        Dict mapping version number -> path to best.pt
    """
    runs_path = Path(os.path.expanduser(runs_dir))
    if not runs_path.exists():
        return {}

    versions = {}
    for d in runs_path.iterdir():
        if not d.is_dir() or not d.name.startswith("YOLO_v"):
            continue
        try:
            v = int(d.name.split("YOLO_v")[1])
        except (ValueError, IndexError):
            continue
        best_pt = d / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = d / "best.pt"
        if best_pt.exists():
            versions[v] = str(best_pt)

    return versions


def select_yolo_model(
    runs_dir: str = DEFAULT_RUNS_DIR,
    allow_custom_path: bool = True,
) -> Optional[str]:
    """Interactive model selector that lists available YOLO versions and lets the user choose.

    Scans runs_dir for YOLO_v* folders, displays them, and suggests the most
    recent version as default. The user can press Enter to accept or type a
    different version number. If the user picks a different version, it confirms
    the choice before proceeding.

    Args:
        runs_dir: Directory containing YOLO_v* version folders.
        allow_custom_path: If True, offer a custom path option when no versions
            are found or the user declines all options.

    Returns:
        Path to the selected model weights, or None if the user cancels.
    """
    versions = find_yolo_versions(runs_dir)

    if not versions:
        runs_path = Path(os.path.expanduser(runs_dir))
        print(f"\n{YELLOW}No YOLO versions found in {runs_path}{RESET}")
        if allow_custom_path:
            custom = input(f"{GREEN}Enter path to model weights (.pt or .onnx): {RESET}").strip()
            return custom if custom else None
        return None

    sorted_versions = sorted(versions.keys())
    latest = sorted_versions[-1]

    # Display available versions
    print(f"\nAvailable YOLO versions in {os.path.expanduser(runs_dir)}:")
    for v in sorted_versions:
        marker = " (latest)" if v == latest else ""
        print(f"  YOLO_v{v}{marker}")
    print()

    # Suggest the latest version
    while True:
        user_input = input(
            f"{GREEN}Use YOLO_v{latest}? [Enter = yes / version number / 'c' for custom path]: {RESET}"
        ).strip()

        if user_input == "":
            # Accept the latest
            print(f"Selected: YOLO_v{latest}")
            return versions[latest]

        if user_input.lower() == "c":
            if allow_custom_path:
                custom = input(f"{GREEN}Enter path to model weights (.pt or .onnx): {RESET}").strip()
                return custom if custom else None
            continue

        # User typed a version number
        try:
            requested = int(user_input)
        except ValueError:
            print(f"{RED}Invalid input. Enter a version number, press Enter, or type 'c'.{RESET}")
            continue

        if requested not in versions:
            print(f"{RED}YOLO_v{requested} not found.{RESET}")
            continue

        # Confirm the chosen version
        confirm = input(
            f"{GREEN}Use YOLO_v{requested}? [Enter = yes / 'n' to go back]: {RESET}"
        ).strip().lower()

        if confirm in ("", "y", "yes"):
            print(f"Selected: YOLO_v{requested}")
            return versions[requested]
        # Otherwise loop back to the main prompt
