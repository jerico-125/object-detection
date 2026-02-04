#!/usr/bin/env python3
"""
Interactive image viewer for reviewing and managing images using matplotlib.
Display images one by one and choose to keep or delete each image.

Usage: python interactive_image_viewer_mpl.py <image_directory> [options]

Examples:
    python interactive_image_viewer_mpl.py ./images/
    python interactive_image_viewer_mpl.py ./sampled_frames/ --extensions jpg png
    python interactive_image_viewer_mpl.py ./dataset/ --move-to-trash
"""

import os
import sys
import argparse
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


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

        # Matplotlib figure and axis
        self.fig = None
        self.ax = None

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

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load an image and convert from BGR to RGB."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image: {image_path}")
            return None

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

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

    def _display_image(self, image_path: Path):
        """Display an image with matplotlib."""
        img = self._load_image(image_path)
        if img is None:
            return False

        # Get status for this image
        status = self.image_status.get(str(image_path), None)

        # Clear previous image
        self.ax.clear()

        # Display image
        self.ax.imshow(img)
        self.ax.axis('off')

        # Add border based on status
        if status == 'keep':
            border_color = 'green'
            status_text = "KEEP"
            for spine in self.ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(10)
                spine.set_visible(True)
        elif status == 'delete':
            border_color = 'red'
            status_text = "DELETE"
            for spine in self.ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(10)
                spine.set_visible(True)
        else:
            status_text = "Not reviewed"
            border_color = 'gray'

        # Create title with info
        title_lines = [
            f"Image {self.current_index + 1}/{self.total_images}",
            f"File: {image_path.name}",
            f"Status: {status_text}"
        ]
        title = '\n'.join(title_lines)

        if status == 'keep':
            self.ax.set_title(title, fontsize=12, color='green', weight='bold', pad=20)
        elif status == 'delete':
            self.ax.set_title(title, fontsize=12, color='red', weight='bold', pad=20)
        else:
            self.ax.set_title(title, fontsize=12, pad=20)

        # Add instructions at the bottom
        instruction_text = "D=Keep | A=Delete | W=Previous | S=Next | Q=Quit"
        self.fig.text(0.5, 0.02, instruction_text, ha='center', fontsize=11,
                     color='green', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        plt.tight_layout()
        self.fig.canvas.draw()
        return True

    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key is None:
            return

        current_path = self.image_paths[self.current_index]
        key = event.key.lower()

        # D key = Keep image and move to next
        if key == 'd':
            # Update status if not already marked as keep
            if self.image_status.get(str(current_path)) != 'keep':
                # If it was previously marked as delete, decrement delete count
                if self.image_status.get(str(current_path)) == 'delete':
                    self.deleted_count -= 1
                self.image_status[str(current_path)] = 'keep'
                self.kept_count += 1
                print(f"[{self.current_index + 1}/{self.total_images}] Marked KEEP: {current_path.name}")
            else:
                print(f"[{self.current_index + 1}/{self.total_images}] Already marked KEEP: {current_path.name}")

            self.current_index += 1
            if self.current_index >= len(self.image_paths):
                print("\nReached end of review.")
                plt.close(self.fig)
                self._process_images()
                self._show_summary()
            else:
                self._display_image(self.image_paths[self.current_index])

        # A key = Delete image and move to next
        elif key == 'a':
            # Update status if not already marked as delete
            if self.image_status.get(str(current_path)) != 'delete':
                # If it was previously marked as keep, decrement keep count
                if self.image_status.get(str(current_path)) == 'keep':
                    self.kept_count -= 1
                self.image_status[str(current_path)] = 'delete'
                self.deleted_count += 1
                print(f"[{self.current_index + 1}/{self.total_images}] Marked DELETE: {current_path.name}")
            else:
                print(f"[{self.current_index + 1}/{self.total_images}] Already marked DELETE: {current_path.name}")

            self.current_index += 1
            if self.current_index >= len(self.image_paths):
                print("\nReached end of review.")
                plt.close(self.fig)
                self._process_images()
                self._show_summary()
            else:
                self._display_image(self.image_paths[self.current_index])

        # W key = Go back to previous image
        elif key == 'w':
            if self.current_index > 0:
                self.current_index -= 1
                print(f"Going back to previous image...")
                self._display_image(self.image_paths[self.current_index])
            else:
                print("Already at first image")

        # S key = Next image (skip without keeping)
        elif key == 's':
            print(f"[{self.current_index + 1}/{self.total_images}] Skipped to next: {current_path.name}")
            self.current_index += 1
            if self.current_index >= len(self.image_paths):
                print("\nReached end of review.")
                plt.close(self.fig)
                self._process_images()
                self._show_summary()
            else:
                self._display_image(self.image_paths[self.current_index])

        # Q or ESC = Quit
        elif key == 'q' or key == 'escape':
            print("\nQuitting review...")
            plt.close(self.fig)
            self._process_images()
            self._show_summary()

    def _on_close(self, event):
        """Handle window close event."""
        print("\nWindow closed. Processing images...")
        self._process_images()
        self._show_summary()

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
        print("Controls: D=Keep | A=Delete | W=Previous | S=Next | Q=Quit\n")

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.manager.set_window_title("Interactive Image Viewer - Review Images")

        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # Display first image
        self.current_index = 0
        if self._display_image(self.image_paths[self.current_index]):
            try:
                plt.show()
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                self._process_images()
                self._show_summary()
        else:
            print("Failed to display first image")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive image viewer for reviewing and managing images (matplotlib version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review all images in a directory
  python interactive_image_viewer_mpl.py ./images/

  # Review only specific image types
  python interactive_image_viewer_mpl.py ./frames/ --extensions jpg png

  # Move deleted images to trash instead of permanent deletion
  python interactive_image_viewer_mpl.py ./dataset/ --move-to-trash

  # Use custom trash directory
  python interactive_image_viewer_mpl.py ./images/ --move-to-trash --trash-dir ./backup/deleted/

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
