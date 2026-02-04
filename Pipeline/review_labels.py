#!/usr/bin/env python3
"""
Step 6: Review Labels

Interactive viewer for reviewing labeled images with annotation overlays.
Supports JSON (LabelMe), TXT (YOLO), and XML (Pascal VOC) label formats.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

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
# LABEL FORMAT DEFINITIONS
# ============================================================================

LABEL_FORMATS = {
    'json': {
        'name': 'JSON (LabelMe/X-AnyLabeling)',
        'extension': '.json',
        'description': 'LabelMe format with shapes and imagePath'
    },
    'txt': {
        'name': 'TXT (YOLO)',
        'extension': '.txt',
        'description': 'YOLO format: class x_center y_center width height'
    },
    'xml': {
        'name': 'XML (Pascal VOC)',
        'extension': '.xml',
        'description': 'Pascal VOC format with bndbox coordinates'
    },
}


# ============================================================================
# LABEL FORMAT DETECTION
# ============================================================================

def detect_label_format(directory: str) -> dict:
    """
    Detect label formats present in a directory.

    Returns:
        Dictionary with format counts and suggested format
    """
    dir_path = Path(directory)
    format_counts = {'json': 0, 'txt': 0, 'xml': 0}

    # Walk through directory and count label files
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith('.json'):
                format_counts['json'] += 1
            elif file_lower.endswith('.txt'):
                # Check if it's likely a YOLO label (not a readme or other txt)
                txt_path = Path(root) / file
                try:
                    with open(txt_path, 'r') as f:
                        first_line = f.readline().strip()
                        # YOLO format: class_id x y w h (5 numbers)
                        parts = first_line.split()
                        if len(parts) == 5 and all(p.replace('.', '').replace('-', '').isdigit() for p in parts):
                            format_counts['txt'] += 1
                except:
                    pass
            elif file_lower.endswith('.xml'):
                format_counts['xml'] += 1

    # Determine suggested format
    suggested = None
    max_count = 0
    for fmt, count in format_counts.items():
        if count > max_count:
            max_count = count
            suggested = fmt

    return {
        'counts': format_counts,
        'suggested': suggested,
        'total': sum(format_counts.values())
    }


# ============================================================================
# ANNOTATION DRAWING UTILITIES
# ============================================================================

COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 128),
    (0, 128, 0), (128, 128, 0),
]


def get_color_for_label(label: str, label_colors: dict) -> tuple:
    if label not in label_colors:
        color_idx = len(label_colors) % len(COLORS)
        label_colors[label] = COLORS[color_idx]
    return label_colors[label]


def load_label_file(label_path: Path, label_format: str, image_shape: tuple) -> dict:
    """
    Load label file and convert to unified format for drawing.

    Args:
        label_path: Path to label file
        label_format: Format type ('json', 'txt', 'xml')
        image_shape: (height, width) of the image for YOLO conversion

    Returns:
        Dictionary with 'shapes' list containing label info
    """
    if label_format == 'json':
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    elif label_format == 'txt':
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        height, width = image_shape[:2]
        shapes = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height

                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2

                    shapes.append({
                        'label': f'class_{class_id}',
                        'points': [[x1, y1], [x2, y2]],
                        'shape_type': 'rectangle'
                    })
        return {'shapes': shapes}

    elif label_format == 'xml':
        # Pascal VOC format using ElementTree
        shapes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        # Find all objects
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            label = name_elem.text if name_elem is not None else 'unknown'

            # Get bounding box
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')

                if all([xmin_elem is not None, ymin_elem is not None,
                        xmax_elem is not None, ymax_elem is not None]):
                    shapes.append({
                        'label': label,
                        'points': [
                            [float(xmin_elem.text), float(ymin_elem.text)],
                            [float(xmax_elem.text), float(ymax_elem.text)]
                        ],
                        'shape_type': 'rectangle'
                    })
        return {'shapes': shapes}

    return {'shapes': []}


def draw_annotations(image: np.ndarray, label_data: dict, label_colors: dict) -> np.ndarray:
    result = image.copy()
    shapes = label_data.get('shapes', [])

    for shape in shapes:
        label = shape.get('label', 'unknown')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'rectangle')

        color = get_color_for_label(label, label_colors)

        if shape_type == 'rectangle' and len(points) >= 2:
            if len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
            else:
                xs = [int(p[0]) for p in points]
                ys = [int(p[1]) for p in points]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)

        elif shape_type == 'polygon' and len(points) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, color, 2)

            x1, y1 = int(points[0][0]), int(points[0][1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, label, (x1, y1 - 5), font, 0.6, color, 2)

    return result


def draw_legend(image: np.ndarray, label_colors: dict) -> np.ndarray:
    result = image.copy()
    y_offset = 30
    for label, color in label_colors.items():
        cv2.rectangle(result, (10, y_offset - 15), (30, y_offset), color, -1)
        cv2.putText(result, label, (35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    return result


def draw_info(image: np.ndarray, info_text: str) -> np.ndarray:
    result = image.copy()
    h, w = result.shape[:2]

    overlay = result.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

    cv2.putText(result, info_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return result


def find_matching_pairs(image_dir: Path, label_dir: Path, label_format: str = 'json') -> list:
    """Find matching image-label pairs based on label format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    label_ext = LABEL_FORMATS.get(label_format, {}).get('extension', '.json')

    pairs = []
    image_files = sorted([f for f in image_dir.iterdir()
                         if f.suffix.lower() in image_extensions])

    for image_path in image_files:
        label_name = image_path.stem + label_ext
        label_path = label_dir / label_name

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            pairs.append((image_path, None))

    return pairs


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def review_labels(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Step 6: Review labeled images."""

    # Determine input directory
    if from_previous_step and config.get("review_input_dir"):
        base_dir = config.get("review_input_dir")
    else:
        base_dir = config.get("consolidated_output_dir", "./consolidated")

    # Ask user for the directory to review
    input_dir = prompt_with_directory_options(
        "Enter the directory containing images and labels to review",
        base_dir,
        ""
    )
    if not input_dir:
        print(f"{RED}No directory provided.{RESET}")
        return False

    if not os.path.exists(input_dir):
        print(f"{RED}Error: Directory '{input_dir}' does not exist.{RESET}")
        return False

    base_path = Path(input_dir)
    review_output_dir = config.get("review_output_dir")

    # Check for Image/Label folder structure or flat structure
    image_dir = base_path / "Image"
    label_dir = base_path / "Label"

    if image_dir.exists() and label_dir.exists():
        print(f"Found Image/Label folder structure in: {base_path}")
    else:
        # Flat structure - images and labels in the same folder
        image_dir = base_path
        label_dir = base_path
        print(f"Using flat folder structure: {base_path}")

    # Detect or use stored label format
    label_format = config.get("label_format")
    if not label_format:
        print("\nDetecting label formats...")
        detection = detect_label_format(str(label_dir))

        print("\nLabel files found:")
        for fmt, count in detection['counts'].items():
            fmt_info = LABEL_FORMATS[fmt]
            print(f"  {fmt_info['name']}: {count} files")

        if detection['total'] == 0:
            print("\nNo label files detected. Will show images without labels.")
            label_format = 'json'  # default
        else:
            # Show format selection menu
            print("\nSelect label format to review:")
            print(f"  1. JSON (LabelMe/X-AnyLabeling) - {detection['counts']['json']} files")
            print(f"  2. TXT (YOLO) - {detection['counts']['txt']} files")
            print(f"  3. XML (Pascal VOC) - {detection['counts']['xml']} files")

            suggested_num = {'json': 1, 'txt': 2, 'xml': 3}.get(detection['suggested'], 1)
            print(f"\nSuggested: Option {suggested_num} ({LABEL_FORMATS[detection['suggested']]['name']})")

            format_choice = input(f"{GREEN}Enter choice (1-3) [default: {suggested_num}]: {RESET}").strip()
            if not format_choice:
                format_choice = str(suggested_num)

            format_map = {'1': 'json', '2': 'txt', '3': 'xml'}
            label_format = format_map.get(format_choice, detection['suggested'] or 'json')

    format_info = LABEL_FORMATS.get(label_format, {})
    print(f"Using label format: {format_info.get('name', label_format)}")

    if review_output_dir:
        Path(review_output_dir).mkdir(parents=True, exist_ok=True)

    pairs = find_matching_pairs(image_dir, label_dir, label_format)

    if not pairs:
        print("No images found.")
        return True

    print(f"\nFound {len(pairs)} images")
    print(f"  With labels: {sum(1 for _, l in pairs if l is not None)}")
    print(f"  Without labels: {sum(1 for _, l in pairs if l is None)}")
    print("\nControls: SPACE/D=Next, A=Previous, S=Save, Q/ESC=Quit\n")

    label_colors = {}
    current_idx = 0
    failed_images = set()  # Track images that failed to load (use set to avoid duplicates)

    window_name = "Label Review - SPACE/D=Next | A=Previous | S=Save | Q=Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_path, label_path = pairs[current_idx]

        image = cv2.imread(str(image_path))
        if image is None:
            failed_images.add(str(image_path))
            # Prevent infinite loop if all images fail
            if len(failed_images) >= len(pairs):
                print(f"{RED}Error: All images failed to load.{RESET}")
                break
            current_idx = (current_idx + 1) % len(pairs)
            continue

        if label_path:
            try:
                label_data = load_label_file(label_path, label_format, image.shape)
                image = draw_annotations(image, label_data, label_colors)
                image = draw_legend(image, label_colors)
                label_status = f"Label: {label_path.name}"
            except Exception as e:
                label_status = f"Error loading label: {e}"
        else:
            label_status = "No label file"

        info_text = f"[{current_idx + 1}/{len(pairs)}] {image_path.name} | {label_status}"
        image = draw_info(image, info_text)

        h, w = image.shape[:2]
        cv2.resizeWindow(window_name, min(1400, w), min(900, h))

        cv2.imshow(window_name, image)

        window_closed = False
        while True:
            key = cv2.waitKey(100) & 0xFF  # 100ms timeout to check window state

            # Check if window was closed via X button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    window_closed = True
                    break
            except cv2.error:
                # Window was destroyed between check and operation
                window_closed = True
                break

            if key == 255:  # No key pressed (timeout)
                continue

            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') or key == 83 or key == ord('d'):
                current_idx = (current_idx + 1) % len(pairs)
                break
            elif key == 81 or key == ord('a'):
                current_idx = (current_idx - 1) % len(pairs)
                break
            elif key == ord('s'):
                if review_output_dir:
                    save_path = Path(review_output_dir) / f"review_{image_path.name}"
                    cv2.imwrite(str(save_path), image)
                    print(f"Saved: {save_path}")
                # Don't break - allow continuing after save

        # Check if we should exit the main loop (window closed or quit pressed)
        if window_closed or key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

    # Show summary of any failed images
    if failed_images:
        failed_list = sorted(failed_images)  # Convert set to sorted list for display
        print(f"\n{RED}Warning: {len(failed_list)} image(s) failed to load:{RESET}")
        for failed_path in failed_list[:10]:  # Show first 10
            print(f"  - {failed_path}")
        if len(failed_list) > 10:
            print(f"  ... and {len(failed_list) - 10} more")

    print("\nReview complete.")
    return True
