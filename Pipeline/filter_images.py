#!/usr/bin/env python3
"""
Step 2: Interactive Image Filtering

Interactive viewer for reviewing and filtering images across multiple folders.
Uses OpenCV for display with keyboard controls (D=Keep, A=Delete, W/S=Navigate, Q=Quit).
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any

import cv2

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
# INTERACTIVE IMAGE VIEWER
# ============================================================================

class RecursiveInteractiveImageViewer:
    """Interactive viewer for reviewing images across multiple folders."""

    def __init__(self, parent_dir: str, extensions: list = None,
                 move_to_trash: bool = False, trash_dir: str = "./deleted/filtered",
                 keep_dir: str = "./kept_images"):
        self.parent_dir = Path(parent_dir)
        self.extensions = extensions or ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        self.move_to_trash = move_to_trash
        self.trash_dir = Path(trash_dir)
        self.keep_dir = Path(keep_dir)

        self.total_images = 0
        self.kept_count = 0
        self.deleted_count = 0
        self.current_index = 0

        self.image_paths = self._load_image_paths_recursive()
        self.total_images = len(self.image_paths)
        self.image_status: Dict[str, str] = {}

        if self.move_to_trash:
            self.trash_dir.mkdir(parents=True, exist_ok=True)
        self.keep_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_paths_recursive(self) -> list:
        if not self.parent_dir.exists():
            raise ValueError(f"Directory does not exist: {self.parent_dir}")

        image_paths = []
        for ext in self.extensions:
            image_paths.extend(self.parent_dir.glob(f"**/*.{ext}"))
            image_paths.extend(self.parent_dir.glob(f"**/*.{ext.upper()}"))

        image_paths.sort()
        return image_paths

    def _get_relative_path(self, image_path: Path) -> Path:
        try:
            return image_path.relative_to(self.parent_dir)
        except ValueError:
            return Path(image_path.name)

    def _display_image(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        height, width = img.shape[:2]
        max_width, max_height = 1200, 900
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_display = cv2.resize(img, (new_width, new_height))
        else:
            img_display = img.copy()

        return img_display

    def _add_info_overlay(self, img, image_path: Path):
        img_with_info = img.copy()
        height, width = img_with_info.shape[:2]

        status = self.image_status.get(str(image_path), None)
        relative_path = self._get_relative_path(image_path)

        border_thickness = 15
        if status == 'keep':
            cv2.rectangle(img_with_info, (0, 0), (width, height), (0, 255, 0), border_thickness)
        elif status == 'delete':
            cv2.rectangle(img_with_info, (0, 0), (width, height), (0, 0, 255), border_thickness)

        overlay = img_with_info.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_with_info, 0.4, 0, img_with_info)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)

        progress_text = f"Image {self.current_index + 1}/{self.total_images}"
        cv2.putText(img_with_info, progress_text, (10, 25), font, font_scale, color, thickness)

        folder_text = f"Folder: {relative_path.parent}"
        cv2.putText(img_with_info, folder_text, (10, 55), font, font_scale, (200, 200, 255), thickness)

        filename_text = f"File: {image_path.name}"
        cv2.putText(img_with_info, filename_text, (10, 85), font, font_scale, color, thickness)

        if status == 'keep':
            status_text = "Status: KEEP"
            status_color = (0, 255, 0)
        elif status == 'delete':
            status_text = "Status: DELETE"
            status_color = (0, 0, 255)
        else:
            status_text = "Status: Not reviewed"
            status_color = (128, 128, 128)

        cv2.putText(img_with_info, status_text, (10, 120), font, font_scale, status_color, thickness)

        overlay2 = img_with_info.copy()
        cv2.rectangle(overlay2, (0, height - 60), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, img_with_info, 0.4, 0, img_with_info)

        instruction_text = "D=Keep | A=Delete | W=Previous | S=Next | Q=Quit"
        text_size = cv2.getTextSize(instruction_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(img_with_info, instruction_text, (text_x, height - 20),
                   font, font_scale, (0, 255, 0), thickness)

        return img_with_info

    def _delete_image(self, image_path: Path) -> bool:
        try:
            relative_path = self._get_relative_path(image_path)

            if self.move_to_trash:
                trash_path = self.trash_dir / relative_path
                trash_path.parent.mkdir(parents=True, exist_ok=True)

                counter = 1
                original_trash_path = trash_path
                while trash_path.exists():
                    trash_path = original_trash_path.parent / f"{original_trash_path.stem}_{counter}{original_trash_path.suffix}"
                    counter += 1
                shutil.move(str(image_path), str(trash_path))
            else:
                image_path.unlink()
            return True
        except Exception as e:
            print(f"{RED}Error deleting {image_path}: {e}{RESET}")
            return False

    def _keep_image(self, image_path: Path) -> bool:
        try:
            relative_path = self._get_relative_path(image_path)
            keep_path = self.keep_dir / relative_path
            keep_path.parent.mkdir(parents=True, exist_ok=True)

            counter = 1
            original_keep_path = keep_path
            while keep_path.exists():
                keep_path = original_keep_path.parent / f"{original_keep_path.stem}_{counter}{original_keep_path.suffix}"
                counter += 1
            shutil.move(str(image_path), str(keep_path))
            return True
        except Exception as e:
            print(f"{RED}Error moving {image_path}: {e}{RESET}")
            return False

    def _process_images(self):
        if self.kept_count > 0:
            print(f"\nProcessing {self.kept_count} images marked to keep...")
            kept_successfully = 0
            for image_path_str, status in self.image_status.items():
                if status == 'keep':
                    image_path = Path(image_path_str)
                    if image_path.exists() and self._keep_image(image_path):
                        kept_successfully += 1
            print(f"Successfully moved {kept_successfully}/{self.kept_count} kept images.")

        if self.deleted_count > 0:
            print(f"\nProcessing {self.deleted_count} images marked for deletion...")
            deleted_successfully = 0
            for image_path_str, status in self.image_status.items():
                if status == 'delete':
                    image_path = Path(image_path_str)
                    if image_path.exists() and self._delete_image(image_path):
                        deleted_successfully += 1
            print(f"Successfully processed {deleted_successfully}/{self.deleted_count} deletions.")

    def _show_summary(self) -> None:
        print("\n" + "=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)
        print(f"Parent directory: {self.parent_dir}")
        print(f"Total images found: {self.total_images}")
        print(f"Images kept: {self.kept_count}")
        print(f"Images deleted: {self.deleted_count}")

        if self.kept_count > 0:
            print(f"\nKept images moved to: {self.keep_dir}")
        print("=" * 60)

    def run(self):
        if not self.image_paths:
            print(f"No images found in {self.parent_dir}")
            return

        print(f"Found {self.total_images} images to review")
        print(f"Kept images will be moved to: {self.keep_dir}")
        print("\nStarting review...\n")

        window_name = "Image Filtering - D=Keep | A=Delete | W/S=Navigate | Q=Quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self.current_index = 0

        try:
            while self.current_index < len(self.image_paths):
                current_path = self.image_paths[self.current_index]

                img = self._display_image(current_path)
                if img is None:
                    self.current_index += 1
                    continue

                img_with_info = self._add_info_overlay(img, current_path)
                cv2.imshow(window_name, img_with_info)

                while True:
                    key = cv2.waitKey(100) & 0xFF  # 100ms timeout to check window state

                    # Check if window was closed via X button
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            cv2.destroyAllWindows()
                            self._process_images()
                            self._show_summary()
                            return
                    except cv2.error:
                        # Window was destroyed between check and operation
                        self._process_images()
                        self._show_summary()
                        return

                    if key == 255:  # No key pressed (timeout)
                        continue

                    if key == ord('d') or key == ord('D'):
                        if self.image_status.get(str(current_path)) != 'keep':
                            if self.image_status.get(str(current_path)) == 'delete':
                                self.deleted_count -= 1
                            self.image_status[str(current_path)] = 'keep'
                            self.kept_count += 1
                        self.current_index += 1
                        break

                    elif key == ord('a') or key == ord('A'):
                        if self.image_status.get(str(current_path)) != 'delete':
                            if self.image_status.get(str(current_path)) == 'keep':
                                self.kept_count -= 1
                            self.image_status[str(current_path)] = 'delete'
                            self.deleted_count += 1
                        self.current_index += 1
                        break

                    elif key == ord('w') or key == ord('W'):
                        if self.current_index > 0:
                            self.current_index -= 1
                            break

                    elif key == ord('s') or key == ord('S'):
                        self.current_index += 1
                        break

                    elif key == ord('q') or key == ord('Q') or key == 27:
                        cv2.destroyAllWindows()
                        self._process_images()
                        self._show_summary()
                        return

            cv2.destroyAllWindows()
            self._process_images()
            self._show_summary()

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            self._process_images()
            self._show_summary()


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def filter_images(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Step 2: Interactive image filtering."""

    # If continuing from previous step, use extracted_frames_dir; otherwise ask
    if from_previous_step:
        input_dir = config.get("extracted_frames_dir", "./extracted_frames")
        video_name = config.get("video_name", "")
    else:
        # Try to get video name from config (set by Step 1 or derived from video_path)
        video_name = config.get("video_name", "")
        if not video_name and config.get("video_path"):
            video_name = Path(config.get("video_path")).stem

        base_dir = config.get("filter_input_dir", "./extracted_frames")
        input_dir = prompt_with_directory_options(
            "Enter the directory containing images to filter",
            base_dir,
            video_name
        )
        if not input_dir:
            print(f"{RED}No directory provided.{RESET}")
            return False
        # Extract video name from input directory if possible (e.g., ./extracted_frames/video_name)
        video_name = Path(input_dir).name
        config["video_name"] = video_name

    # Create output directories with video_name subfolder
    base_keep_dir = config.get("kept_images_dir", "./kept_images")
    base_trash_dir = config.get("deleted_images_dir", "./deleted/filtered")

    if video_name:
        keep_dir = f"{base_keep_dir}/{video_name}"
        trash_dir = f"{base_trash_dir}/{video_name}"
    else:
        keep_dir = base_keep_dir
        trash_dir = base_trash_dir

    move_to_trash = config.get("move_to_trash", True)
    extensions = config.get("image_extensions", ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])

    if not os.path.exists(input_dir):
        print(f"{RED}Error: Input directory '{input_dir}' does not exist.{RESET}")
        new_dir = input(f"{GREEN}Enter the directory containing images to filter: {RESET}").strip()
        if new_dir and os.path.exists(new_dir):
            input_dir = new_dir
        else:
            print(f"{RED}Invalid directory.{RESET}")
            return False

    print(f"Input directory: {input_dir}")
    print(f"Kept images will be saved to: {keep_dir}")
    print(f"Deleted images will be saved to: {trash_dir}")

    try:
        viewer = RecursiveInteractiveImageViewer(
            parent_dir=input_dir,
            extensions=extensions,
            move_to_trash=move_to_trash,
            trash_dir=trash_dir,
            keep_dir=keep_dir
        )
        viewer.run()

        # Update config for next step (anonymization comes before labeling)
        config["anonymize_input_dir"] = keep_dir

        return True

    except Exception as e:
        print(f"{RED}Error during filtering: {e}{RESET}")
        return False
