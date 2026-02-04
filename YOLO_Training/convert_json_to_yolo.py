#!/usr/bin/env python3
"""
JSON to YOLO Format Converter

Converts JSON annotation files (LabelMe / X-AnyLabeling format) to YOLO format.
YOLO format: class_id x_center y_center width height (all normalized 0-1)

Usage:
    python convert_json_to_yolo.py --input_dir /path/to/dataset --output_dir /path/to/output
    python convert_json_to_yolo.py  # Interactive mode - prompts for inputs
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image


def load_classes_from_file(classes_file: str) -> Dict[str, int]:
    """Load class mapping from classes.txt file."""
    classes = {}
    with open(classes_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                # Format: "id: class_name" or "category_id: class_name"
                parts = line.split(':', 1)
                class_id = parts[0].strip()
                class_name = parts[1].strip()
                # Use 0-based indexing for YOLO
                if class_id.isdigit():
                    classes[class_name] = int(class_id) - 1  # Convert to 0-based
                else:
                    classes[class_id] = len(classes)
                classes[class_name.lower()] = classes.get(class_name, len(classes))
    return classes


def build_class_mapping(label_dir: str, classes_file: Optional[str] = None) -> Tuple[Dict[str, int], List[str]]:
    """
    Build class mapping from labels or classes.txt file.
    Returns: (class_to_id dict, class_names list)
    """
    if classes_file and os.path.exists(classes_file):
        print(f"Loading classes from: {classes_file}")
        raw_classes = load_classes_from_file(classes_file)
        # Rebuild with proper 0-based indexing
        unique_names = sorted(set(k for k in raw_classes.keys() if not k.replace('_', '').isalnum() or not k[0].isdigit()))
        class_names = list(unique_names) if unique_names else list(set(raw_classes.keys()))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        return class_to_id, class_names

    # Scan all JSON files to extract unique classes
    print("Scanning JSON files to extract class names...")
    class_names_set = set()

    for json_file in Path(label_dir).glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = extract_annotations(data)
            for ann in annotations:
                if 'category_name' in ann:
                    class_names_set.add(ann['category_name'])
                elif 'category_id' in ann:
                    class_names_set.add(str(ann['category_id']))
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")

    class_names = sorted(list(class_names_set))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    return class_to_id, class_names


def extract_annotations(data: dict) -> List[dict]:
    """Extract annotation list from various JSON structures."""
    # LabelMe / X-AnyLabeling format
    if 'shapes' in data:
        boxes = []
        # Get image dimensions from the JSON if available
        img_width = data.get('imageWidth', 0)
        img_height = data.get('imageHeight', 0)

        for shape in data['shapes']:
            if shape.get('shape_type') == 'rectangle':
                points = shape['points']

                if len(points) == 2:
                    # 2-point format: [top-left, bottom-right]
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                elif len(points) == 4:
                    # 4-point format: [top-left, top-right, bottom-right, bottom-left]
                    # Extract all x and y coordinates
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                else:
                    print(f"Warning: Unexpected number of points ({len(points)}) for rectangle")
                    continue

                boxes.append({
                    'box': {'x': min(x1, x2), 'y': min(y1, y2),
                            'w': abs(x2 - x1), 'h': abs(y2 - y1)},
                    'category_name': shape.get('label', 'unknown'),
                    'img_width': img_width,
                    'img_height': img_height
                })
        return boxes

    # Simple format with direct 'annotations' list
    if 'annotations' in data:
        return data['annotations']

    # COCO-like format
    if 'images' in data and 'annotations' in data:
        return data['annotations']

    return []


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_to_yolo_format(box: dict, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert box coordinates to YOLO format (normalized x_center, y_center, width, height).
    """
    if isinstance(box, dict):
        x = box.get('x', 0)
        y = box.get('y', 0)
        w = box.get('w', box.get('width', 0))
        h = box.get('h', box.get('height', 0))
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        x, y, w, h = box[:4]
    else:
        raise ValueError(f"Unknown box format: {box}")

    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize width and height
    norm_width = w / img_width
    norm_height = h / img_height

    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))

    return x_center, y_center, norm_width, norm_height


def process_single_image_json(json_file: str, image_dir: str, output_label_dir: str,
                               class_to_id: Dict[str, int], output_image_dir: Optional[str] = None) -> int:
    """Process a JSON file that corresponds to a single image."""
    json_path = Path(json_file)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Try to get image path from JSON (LabelMe format)
    image_name = data.get('imagePath', json_path.stem)
    # Remove extension if present to normalize
    image_stem = Path(image_name).stem

    # Find corresponding image
    image_path = None

    # First try the exact imagePath from JSON
    if 'imagePath' in data:
        candidate = Path(image_dir) / data['imagePath']
        if candidate.exists():
            image_path = candidate

    # If not found, search with different extensions
    if not image_path:
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.PNG', '.JPG', '.JPEG']:
            candidate = Path(image_dir) / f"{image_stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

    if not image_path:
        print(f"Warning: No image found for {json_file}")
        return 0

    # Get image dimensions - prefer from JSON (faster), fallback to reading image
    img_width = data.get('imageWidth', 0)
    img_height = data.get('imageHeight', 0)

    if img_width == 0 or img_height == 0:
        img_width, img_height = get_image_dimensions(str(image_path))

    # Extract annotations
    annotations = extract_annotations(data)

    if not annotations:
        print(f"Warning: No annotations in {json_file}")
        return 0

    # Write YOLO format labels (use image stem, not full name with extension)
    output_label_path = Path(output_label_dir) / f"{image_path.stem}.txt"

    lines = []
    for ann in annotations:
        box = ann.get('box', ann)
        category = ann.get('category_name', ann.get('category_id', 'unknown'))

        if str(category) not in class_to_id and category not in class_to_id:
            print(f"Warning: Unknown class '{category}' in {json_file}")
            continue

        class_id = class_to_id.get(str(category), class_to_id.get(category, 0))

        try:
            x_center, y_center, w, h = convert_to_yolo_format(box, img_width, img_height)
            if w > 0 and h > 0:
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        except Exception as e:
            print(f"Warning: Could not convert box in {json_file}: {e}")

    if lines:
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(lines))

        # Copy image if output_image_dir specified
        if output_image_dir:
            shutil.copy2(image_path, Path(output_image_dir) / image_path.name)

        return 1

    return 0


def convert_dataset(input_dir: str, output_dir: str, train_ratio: float = 0.8,
                    classes_file: Optional[str] = None):
    """
    Convert entire dataset from JSON to YOLO format.

    Args:
        input_dir: Directory containing Image/ and Label/ folders
        output_dir: Output directory for YOLO format dataset
        train_ratio: Ratio of training data (rest goes to validation)
        classes_file: Optional path to classes.txt
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Detect directory structure
    image_dir = input_path / 'Image'
    label_dir = input_path / 'Label'

    if not image_dir.exists():
        image_dir = input_path / 'images'
    if not label_dir.exists():
        label_dir = input_path / 'labels'

    if not image_dir.exists() or not label_dir.exists():
        print(f"Error: Expected 'Image' and 'Label' folders in {input_dir}")
        print(f"Found: {list(input_path.iterdir())}")
        return

    # Look for classes.txt
    if not classes_file:
        for candidate in [input_path / 'classes.txt', label_dir / 'classes.txt']:
            if candidate.exists():
                classes_file = str(candidate)
                break

    # Build class mapping
    class_to_id, class_names = build_class_mapping(str(label_dir), classes_file)
    print(f"\nDetected {len(class_names)} classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # Create output directories
    train_images = output_path / 'images' / 'train'
    train_labels = output_path / 'labels' / 'train'
    val_images = output_path / 'images' / 'val'
    val_labels = output_path / 'labels' / 'val'

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = list(label_dir.glob('*.json'))
    print(f"\nFound {len(json_files)} JSON files")

    # Split into train/val
    import random
    random.shuffle(json_files)
    split_idx = int(len(json_files) * train_ratio)
    train_jsons = json_files[:split_idx]
    val_jsons = json_files[split_idx:]

    print(f"Train: {len(train_jsons)} files, Val: {len(val_jsons)} files")

    # Process training data
    print("\nProcessing training data...")
    train_count = 0
    for json_file in train_jsons:
        count = process_single_image_json(str(json_file), str(image_dir), str(train_labels),
                                          class_to_id, str(train_images))
        train_count += count

    # Process validation data
    print("Processing validation data...")
    val_count = 0
    for json_file in val_jsons:
        count = process_single_image_json(str(json_file), str(image_dir), str(val_labels),
                                          class_to_id, str(val_images))
        val_count += count

    print(f"\nConversion complete!")
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")

    # Save class names
    classes_output = output_path / 'classes.txt'
    with open(classes_output, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"  Classes saved to: {classes_output}")

    # Generate YAML config
    yaml_path = output_path / 'dataset.yaml'
    yaml_content = f"""# YOLO Dataset Configuration
# Auto-generated by convert_json_to_yolo.py

path: {output_path.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"  Dataset YAML saved to: {yaml_path}")

    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description='Convert JSON annotations to YOLO format')
    parser.add_argument('--input_dir', '-i', default=None,
                        help='Input directory containing Image/ and Label/ folders')
    parser.add_argument('--output_dir', '-o', default=None,
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--train_ratio', '-r', type=float, default=None,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--classes_file', '-c', default=None,
                        help='Path to classes.txt file (optional)')

    args = parser.parse_args()

    # Prompt for required arguments if not provided
    input_dir = args.input_dir
    if not input_dir:
        input_dir = input("Input directory (containing Image/ and Label/ folders): ").strip()

    output_dir = args.output_dir
    if not output_dir:
        output_dir = input("Output directory for YOLO format dataset: ").strip()

    train_ratio = args.train_ratio
    if train_ratio is None:
        ratio_input = input("Train/val split ratio [0.8]: ").strip()
        train_ratio = float(ratio_input) if ratio_input else 0.8

    classes_file = args.classes_file
    if not classes_file:
        classes_input = input("Path to classes.txt file (leave blank to auto-detect): ").strip()
        classes_file = classes_input if classes_input else None

    convert_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        classes_file=classes_file
    )


if __name__ == '__main__':
    main()
