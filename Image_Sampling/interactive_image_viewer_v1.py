#!/usr/bin/env python3
"""
Interactive image viewer for reviewing and managing images.
Display images one by one and choose to keep or delete each image.

Usage: python interactive_image_viewer.py <image_directory> [options]

Examples:
    python interactive_image_viewer.py ./images/
    python interactive_image_viewer.py ./sampled_frames/ --extensions jpg png
    python interactive_image_viewer.py ./dataset/ --move-to-trash
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import sys
import argparse
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional


class InteractiveImageViewer:
    """Interactive viewer for reviewing images with keep/delete functionality."""

    def __init__(self, image_dir: str, extensions: List[str] = None,
                 move_to_trash: bool = False, trash_dir: str = "./deleted_images",
                 keep_dir: str = "./kept_images"):
        """
        Initialize the interactive image viewer.

        Args:
            image_dir: Directory containing images to review
            extensions: List of image file extensions to include (e.g., ['jpg', 'png'])
            move_to_trash: If True, move deleted images to trash folder instead of deleting
            trash_dir: Directory to move deleted images (only used if move_to_trash=True)
            keep_dir: Directory to move kept images
        """
        self.image_dir = Path(image_dir)
        self.extensions = extensions or ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        self.move_to_trash = move_to_trash
        self.trash_dir = Path(trash_dir)

        # Include source directory name in keep_dir
        source_dir_name = self.image_dir.name if self.image_dir.name else "images"
        self.keep_dir = Path(keep_dir) / source_dir_name

        # Statistics
        self.total_images = 0
        self.kept_count = 0
        self.deleted_count = 0
        self.current_index = 0

        # Load all image paths
        self.image_paths = self._load_image_paths()
        self.total_images = len(self.image_paths)

        # Track image status: None = not reviewed, 'keep' = marked to keep, 'delete' = marked to delete
        self.image_status = {}

        # Create directories
        if self.move_to_trash:
            self.trash_dir.mkdir(parents=True, exist_ok=True)
        self.keep_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_paths(self) -> List[Path]:
        """Load all image paths from the directory."""
        if not self.image_dir.exists():
            raise ValueError(f"Directory does not exist: {self.image_dir}")

        image_paths = []
        for ext in self.extensions:
            image_paths.extend(self.image_dir.glob(f"*.{ext}"))
            image_paths.extend(self.image_dir.glob(f"*.{ext.upper()}"))

        # Sort by filename
        image_paths.sort()
        return image_paths

    def _display_image(self, image_path: Path) -> Optional[cv2.Mat]:
        """Load and display an image."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image: {image_path}")
            return None

        # Get image dimensions
        height, width = img.shape[:2]

        # Resize if too large (max 1200x900 for display)
        max_width, max_height = 1200, 900
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_display = cv2.resize(img, (new_width, new_height))
        else:
            img_display = img.copy()

        return img_display

    def _add_info_overlay(self, img: cv2.Mat, image_path: Path) -> cv2.Mat:
        """Add information overlay to the image."""
        img_with_info = img.copy()
        height, width = img_with_info.shape[:2]

        # Get status for this image
        status = self.image_status.get(str(image_path), None)

        # Add status indicator border
        border_thickness = 15
        if status == 'keep':
            # Green border for kept images
            cv2.rectangle(img_with_info, (0, 0), (width, height), (0, 255, 0), border_thickness)
        elif status == 'delete':
            # Red border for deleted images
            cv2.rectangle(img_with_info, (0, 0), (width, height), (0, 0, 255), border_thickness)

        # Create a semi-transparent overlay at the top
        overlay = img_with_info.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_with_info, 0.4, 0, img_with_info)

        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)

        # Progress info
        progress_text = f"Image {self.current_index + 1}/{self.total_images}"
        cv2.putText(img_with_info, progress_text, (10, 25), font, font_scale, color, thickness)

        # Filename
        filename_text = f"File: {image_path.name}"
        cv2.putText(img_with_info, filename_text, (10, 55), font, font_scale, color, thickness)

        # Status indicator
        if status == 'keep':
            status_text = "Status: KEEP"
            status_color = (0, 255, 0)  # Green
        elif status == 'delete':
            status_text = "Status: DELETE"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "Status: Not reviewed"
            status_color = (128, 128, 128)  # Gray

        cv2.putText(img_with_info, status_text, (10, 90), font, font_scale, status_color, thickness)

        # Add instruction overlay at the bottom
        overlay2 = img_with_info.copy()
        cv2.rectangle(overlay2, (0, height - 60), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, img_with_info, 0.4, 0, img_with_info)

        # Instructions
        instruction_text = "D=Keep | A=Delete | W=Previous | S=Next | Q=Quit"
        text_size = cv2.getTextSize(instruction_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(img_with_info, instruction_text, (text_x, height - 20),
                   font, font_scale, (0, 255, 0), thickness)

        return img_with_info

    def _delete_image(self, image_path: Path) -> bool:
        """Delete or move image to trash."""
        try:
            if self.move_to_trash:
                # Move to trash directory
                trash_path = self.trash_dir / image_path.name
                # Handle duplicate names
                counter = 1
                while trash_path.exists():
                    trash_path = self.trash_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
                    counter += 1
                shutil.move(str(image_path), str(trash_path))
                print(f"Moved to trash: {trash_path}")
            else:
                # Permanently delete
                image_path.unlink()
                print(f"Deleted: {image_path.name}")
            return True
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")
            return False

    def _keep_image(self, image_path: Path) -> bool:
        """Move image to keep directory."""
        try:
            # Move to keep directory
            keep_path = self.keep_dir / image_path.name
            # Handle duplicate names
            counter = 1
            while keep_path.exists():
                keep_path = self.keep_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
                counter += 1
            shutil.move(str(image_path), str(keep_path))
            print(f"Moved to keep folder: {keep_path}")
            return True
        except Exception as e:
            print(f"Error moving {image_path}: {e}")
            return False

    def _process_images(self):
        """Process all marked images (move kept images and delete/trash deleted images)."""
        # Process kept images
        if self.kept_count > 0:
            print(f"\nProcessing {self.kept_count} images marked to keep...")
            kept_successfully = 0
            for image_path_str, status in self.image_status.items():
                if status == 'keep':
                    image_path = Path(image_path_str)
                    if self._keep_image(image_path):
                        kept_successfully += 1
            print(f"Successfully moved {kept_successfully}/{self.kept_count} kept images.")
        else:
            print("\nNo images marked to keep.")

        # Process deleted images
        if self.deleted_count > 0:
            print(f"\nProcessing {self.deleted_count} images marked for deletion...")
            deleted_successfully = 0
            for image_path_str, status in self.image_status.items():
                if status == 'delete':
                    image_path = Path(image_path_str)
                    if self._delete_image(image_path):
                        deleted_successfully += 1
            print(f"Successfully processed {deleted_successfully}/{self.deleted_count} deletions.")
        else:
            print("\nNo images marked for deletion.")

    def _show_summary(self):
        """Show final summary statistics."""
        print("\n" + "=" * 60)
        print("REVIEW SUMMARY")
        print("=" * 60)
        print(f"Total images reviewed: {self.current_index}")
        print(f"Images kept: {self.kept_count}")
        print(f"Images deleted: {self.deleted_count}")

        if self.total_images > 0:
            kept_percentage = (self.kept_count / self.total_images) * 100
            deleted_percentage = (self.deleted_count / self.total_images) * 100
            print(f"Keep rate: {kept_percentage:.1f}%")
            print(f"Delete rate: {deleted_percentage:.1f}%")

        if self.kept_count > 0:
            print(f"\nKept images moved to: {self.keep_dir}")
        if self.move_to_trash and self.deleted_count > 0:
            print(f"Deleted images moved to: {self.trash_dir}")
        elif not self.move_to_trash and self.deleted_count > 0:
            print(f"Deleted images permanently removed")
        print("=" * 60)

    def run(self):
        """Run the interactive viewer."""
        if not self.image_paths:
            print(f"No images found in {self.image_dir}")
            print(f"Looking for extensions: {', '.join(self.extensions)}")
            return

        print(f"Found {self.total_images} images to review")
        print(f"Image directory: {self.image_dir}")
        print(f"Kept images will be moved to: {self.keep_dir}")
        if self.move_to_trash:
            print(f"Deleted images will be moved to: {self.trash_dir}")
        else:
            print("Warning: Deleted images will be permanently removed!")
        print("\nStarting review...\n")

        window_name = "Interactive Image Viewer - Review Images"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self.current_index = 0

        try:
            while self.current_index < len(self.image_paths):
                current_path = self.image_paths[self.current_index]

                # Display current image
                img = self._display_image(current_path)
                if img is None:
                    # Skip unreadable images
                    self.current_index += 1
                    continue

                # Add info overlay
                img_with_info = self._add_info_overlay(img, current_path)

                # Show image
                cv2.imshow(window_name, img_with_info)

                # Wait for key press
                while True:
                    key = cv2.waitKey(0) & 0xFF

                    # D key = Keep image and move to next
                    if key == ord('d') or key == ord('D'):
                        # Update status if not already marked as keep
                        if self.image_status.get(str(current_path)) != 'keep':
                            # If it was previously marked as delete, decrement delete count
                            if self.image_status.get(str(current_path)) == 'delete':
                                self.deleted_count -= 1
                            # If it's new, increment keep count
                            elif self.image_status.get(str(current_path)) is None:
                                pass  # Will increment below
                            self.image_status[str(current_path)] = 'keep'
                            self.kept_count += 1
                            print(f"[{self.current_index + 1}/{self.total_images}] Marked KEEP: {current_path.name}")
                        else:
                            print(f"[{self.current_index + 1}/{self.total_images}] Already marked KEEP: {current_path.name}")
                        self.current_index += 1
                        break

                    # A key = Delete image and move to next
                    elif key == ord('a') or key == ord('A'):
                        # Update status if not already marked as delete
                        if self.image_status.get(str(current_path)) != 'delete':
                            # If it was previously marked as keep, decrement keep count
                            if self.image_status.get(str(current_path)) == 'keep':
                                self.kept_count -= 1
                            # If it's new, increment delete count
                            elif self.image_status.get(str(current_path)) is None:
                                pass  # Will increment below
                            self.image_status[str(current_path)] = 'delete'
                            self.deleted_count += 1
                            print(f"[{self.current_index + 1}/{self.total_images}] Marked DELETE: {current_path.name}")
                        else:
                            print(f"[{self.current_index + 1}/{self.total_images}] Already marked DELETE: {current_path.name}")
                        self.current_index += 1
                        break

                    # W key = Go back to previous image
                    elif key == ord('w') or key == ord('W'):
                        if self.current_index > 0:
                            self.current_index -= 1
                            print(f"Going back to previous image...")
                            break
                        else:
                            print("Already at first image")

                    # S key = Next image (skip without keeping)
                    elif key == ord('s') or key == ord('S'):
                        print(f"[{self.current_index + 1}/{self.total_images}] Skipped to next: {current_path.name}")
                        self.current_index += 1
                        break

                    # Q or ESC = Quit
                    elif key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                        print("\nQuitting review...")
                        cv2.destroyAllWindows()
                        self._process_images()
                        self._show_summary()
                        return

            # Reached end of images
            print("\nReached end of review.")
            cv2.destroyAllWindows()
            self._process_images()
            self._show_summary()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            cv2.destroyAllWindows()
            self._process_images()
            self._show_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive image viewer for reviewing and managing images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review all images in a directory
  python interactive_image_viewer.py ./images/

  # Review only specific image types
  python interactive_image_viewer.py ./frames/ --extensions jpg png

  # Move deleted images to trash instead of permanent deletion
  python interactive_image_viewer.py ./dataset/ --move-to-trash

  # Use custom trash directory
  python interactive_image_viewer.py ./images/ --move-to-trash --trash-dir ./backup/deleted/

Controls:
  D = Keep image and continue
  A = Delete image
  W = Go back to previous image
  S = Skip to next image
  Q / ESC = Quit review
        """
    )

    parser.add_argument('image_dir', help='Directory containing images to review')
    parser.add_argument('--extensions', '-e', nargs='+',
                       default=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
                       help='Image file extensions to include (default: jpg jpeg png bmp tiff webp)')
    parser.add_argument('--move-to-trash', '-m', action='store_true',
                       help='Move deleted images to trash folder instead of permanent deletion')
    parser.add_argument('--trash-dir', '-t', default='./deleted_images',
                       help='Trash directory for deleted images (default: ./deleted_images)')
    parser.add_argument('--keep-dir', '-k', default='./kept_images',
                       help='Directory to move kept images (default: ./kept_images)')

    args = parser.parse_args()

    try:
        viewer = InteractiveImageViewer(
            image_dir=args.image_dir,
            extensions=args.extensions,
            move_to_trash=args.move_to_trash,
            trash_dir=args.trash_dir,
            keep_dir=args.keep_dir
        )
        viewer.run()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()