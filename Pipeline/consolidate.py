#!/usr/bin/env python3
"""
Step 5: Consolidate Files

Consolidate images and labels from subdirectories into a single organized folder.
Supports JSON (LabelMe), TXT (YOLO), and XML (Pascal VOC) label formats.
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
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

# Supported label formats
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
# IMAGE CONSOLIDATOR
# ============================================================================

class ImageConsolidator:
    """Consolidate images from subdirectories into a single folder."""

    def __init__(self, source_dir: str, output_dir: str = "./consolidated_images",
                 extensions: list = None, copy_files: bool = True,
                 include_labels: bool = False, separate_folders: bool = True,
                 label_format: str = 'json'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.extensions = extensions or ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        self.copy_files = copy_files
        self.include_labels = include_labels
        self.separate_folders = separate_folders
        self.label_format = label_format

        if self.separate_folders and self.include_labels:
            self.image_output_dir = self.output_dir / "Image"
            self.label_output_dir = self.output_dir / "Label"
        else:
            self.image_output_dir = self.output_dir
            self.label_output_dir = self.output_dir

        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.processed_labels = 0
        self.missing_labels = 0

    def _find_all_images(self) -> list:
        if not self.source_dir.exists():
            raise ValueError(f"Directory does not exist: {self.source_dir}")

        image_paths = []
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                if any(file.lower().endswith(f'.{ext}') for ext in self.extensions):
                    image_paths.append(file_path)

        image_paths.sort()
        return image_paths

    def _get_starting_index(self) -> int:
        import re
        if not self.image_output_dir.exists():
            return 1

        max_index = 0
        pattern = re.compile(r'^Image(\d+)\.[a-zA-Z]+$')

        for file in self.image_output_dir.iterdir():
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)

        return max_index + 1

    def _get_new_filename(self, index: int, original_path: Path, is_label: bool = False) -> str:
        if is_label:
            label_ext = LABEL_FORMATS.get(self.label_format, {}).get('extension', '.json')
            return f"Image{index:08d}{label_ext}"
        else:
            extension = original_path.suffix
            return f"Image{index:08d}{extension}"

    def _find_label_file(self, image_path: Path) -> Optional[Path]:
        """Find label file for an image based on the selected format."""
        label_ext = LABEL_FORMATS.get(self.label_format, {}).get('extension', '.json')
        label_path = image_path.with_suffix(label_ext)
        if label_path.exists():
            return label_path
        return None

    def _update_label_json(self, label_path: Path, new_image_filename: str) -> dict:
        """Update JSON label file (LabelMe format)."""
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'imagePath' in data:
            data['imagePath'] = new_image_filename

        return data

    def _update_label_txt(self, label_path: Path, new_image_filename: str) -> str:
        """Update TXT label file (YOLO format) - no changes needed, just read."""
        with open(label_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _update_label_xml(self, label_path: Path, new_image_filename: str) -> ET.Element:
        """Update XML label file (Pascal VOC format)."""
        tree = ET.parse(label_path)
        root = tree.getroot()

        # Update <filename> tag
        filename_elem = root.find('filename')
        if filename_elem is not None:
            filename_elem.text = new_image_filename

        # Update <path> tag if present
        path_elem = root.find('path')
        if path_elem is not None:
            path_elem.text = new_image_filename

        return root

    def _save_label_file(self, data, output_path: Path):
        """Save label file based on format."""
        if self.label_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif self.label_format == 'xml':
            # data is an ElementTree Element
            tree = ET.ElementTree(data)
            tree.write(output_path, encoding='unicode', xml_declaration=True)
        else:
            # TXT is saved as string
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data)

    def _process_label(self, label_path: Path, new_image_filename: str):
        """Process label file based on format."""
        if self.label_format == 'json':
            return self._update_label_json(label_path, new_image_filename)
        elif self.label_format == 'txt':
            return self._update_label_txt(label_path, new_image_filename)
        elif self.label_format == 'xml':
            return self._update_label_xml(label_path, new_image_filename)
        else:
            raise ValueError(f"Unsupported label format: {self.label_format}")

    def consolidate(self) -> None:
        print(f"{CYAN}Scanning directory: {self.source_dir}{RESET}")
        image_paths = self._find_all_images()
        self.total_images = len(image_paths)

        if self.total_images == 0:
            print(f"No images found in {self.source_dir}")
            return

        print(f"{CYAN}Found {self.total_images} images to consolidate{RESET}")
        if self.include_labels:
            format_info = LABEL_FORMATS.get(self.label_format, {})
            print(f"{CYAN}Label format: {format_info.get('name', self.label_format)}{RESET}")

        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        if self.include_labels and self.separate_folders:
            self.label_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"{CYAN}Output: {self.output_dir} (Image/ and Label/ subfolders){RESET}")
        else:
            print(f"{CYAN}Output directory: {self.output_dir}{RESET}")

        start_index = self._get_starting_index()

        print("\nProcessing images...\n")

        for idx, image_path in enumerate(image_paths, start=start_index):
            try:
                new_filename = self._get_new_filename(idx, image_path)
                new_path = self.image_output_dir / new_filename

                label_path = None
                new_label_path = None
                label_data = None

                if self.include_labels:
                    label_path = self._find_label_file(image_path)
                    if label_path:
                        new_label_filename = self._get_new_filename(idx, image_path, is_label=True)
                        new_label_path = self.label_output_dir / new_label_filename

                        # Process label FIRST to validate before moving/copying files
                        try:
                            label_data = self._process_label(label_path, new_filename)
                        except Exception as e:
                            print(f"{RED}Warning: Failed to process label {label_path}: {e}{RESET}")
                            print(f"{RED}  Skipping image-label pair to maintain dataset integrity.{RESET}")
                            self.failed_images += 1
                            continue

                if new_path.exists():
                    self.failed_images += 1
                    continue

                # Now safe to move/copy files - label is already validated
                if self.copy_files:
                    shutil.copy2(str(image_path), str(new_path))
                else:
                    shutil.move(str(image_path), str(new_path))

                self.processed_images += 1

                if self.include_labels and label_path and label_data is not None:
                    self._save_label_file(label_data, new_label_path)

                    if not self.copy_files:
                        label_path.unlink()

                    self.processed_labels += 1
                elif self.include_labels and not label_path:
                    self.missing_labels += 1

                if idx % 10 == 0 or self.total_images <= 20:
                    label_info = " (+label)" if label_path else ""
                    status = f"[{idx}/{self.total_images}] {image_path.name} -> {new_filename}{label_info}"
                    print(f"\r\033[K{status}", end="", flush=True)

            except Exception as e:
                print(f"\r\033[K{RED}Error processing {image_path}: {e}{RESET}")
                self.failed_images += 1

        print()  # Final newline after in-place updates
        self._show_summary()

    def _show_summary(self) -> None:
        print("\n" + "=" * 60)
        print("CONSOLIDATION SUMMARY")
        print("=" * 60)
        print(f"{CYAN}Total images found: {self.total_images}")
        print(f"Successfully processed: {self.processed_images}")
        print(f"Failed: {self.failed_images}")

        if self.include_labels:
            print(f"Label files processed: {self.processed_labels}")
            print(f"Images without labels: {self.missing_labels}")
            format_info = LABEL_FORMATS.get(self.label_format, {})
            print(f"Label format: {format_info.get('name', self.label_format)}")

        if self.processed_images > 0:
            print(f"\nConsolidated images saved to: {self.image_output_dir}")
            if self.include_labels and self.separate_folders:
                print(f"Label files saved to: {self.label_output_dir}")

        print(RESET, end="")
        print("=" * 60)


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def consolidate_files(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Step 5: Consolidate images and labels."""

    # Get video name (from previous steps or derived from video_path)
    video_name = config.get("video_name", "")
    if not video_name and config.get("video_path"):
        video_name = Path(config.get("video_path")).stem

    # Ask for input directory (labels are typically saved with images)
    base_dir = config.get("labeling_input_dir", "./kept_images")
    first_input = prompt_with_directory_options(
        "Enter the directory containing images and labels to consolidate",
        base_dir,
        video_name
    )
    if not first_input:
        print(f"{RED}No directory provided.{RESET}")
        return False

    if not os.path.exists(first_input):
        print(f"{RED}Error: Input directory '{first_input}' does not exist.{RESET}")
        return False

    input_dirs = [first_input]

    output_dir = config.get("consolidated_output_dir", "./consolidated")
    include_labels = config.get("include_labels", True)
    copy_files = config.get("copy_files", True)
    separate_folders = config.get("separate_folders", True)
    extensions = config.get("image_extensions", ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])

    print(f"Input directory: {first_input}")

    # Detect label format
    label_format = config.get("label_format", "json")
    skip_format_prompt = config.get("skip_format_prompt", False)

    if include_labels:
        if skip_format_prompt:
            # Use the configured format directly without prompting
            print(f"\nUsing label format: {LABEL_FORMATS[label_format]['name']}")
        else:
            print("\nDetecting label formats...")
            detection = detect_label_format(first_input)

            if detection['total'] == 0:
                print("\nLabel files found: none")
                print("\nNo label files detected.")
                include_choice = input(f"{GREEN}Continue without labels? (y/n): {RESET}").strip().lower()
                if include_choice != 'y':
                    return False
                include_labels = False
            else:
                # Show format selection menu
                print("\nSelect label format to process:")
                print(f"  1. JSON (LabelMe/X-AnyLabeling) - {detection['counts']['json']} files")
                print(f"  2. TXT (YOLO) - {detection['counts']['txt']} files")
                print(f"  3. XML (Pascal VOC) - {detection['counts']['xml']} files")

                # Suggest the format with most files
                suggested_num = {'json': 1, 'txt': 2, 'xml': 3}.get(detection['suggested'], 1)
                print(f"\nSuggested: Option {suggested_num} ({LABEL_FORMATS[detection['suggested']]['name']})")

                format_choice = input(f"{GREEN}Enter choice (1-3) [default: {suggested_num}]: {RESET}").strip()
                if not format_choice:
                    format_choice = str(suggested_num)

                format_map = {'1': 'json', '2': 'txt', '3': 'xml'}
                label_format = format_map.get(format_choice, detection['suggested'] or 'json')

                print(f"\nUsing label format: {LABEL_FORMATS[label_format]['name']}")

    # Confirm input and output directories before proceeding
    while True:
        # Warn if output directory already exists and has contents
        output_path = Path(output_dir)
        if output_path.exists():
            existing_images = [
                f for f in output_path.rglob('*')
                if f.is_file() and any(f.suffix.lower() == f'.{ext}' for ext in extensions)
            ]
            existing_total = len(list(f for f in output_path.rglob('*') if f.is_file()))
            if existing_total:
                print()
                print(f"{RED}WARNING: Output directory '{output_dir}' already exists.{RESET}")
                print(f"  Existing files: {existing_total} ({len(existing_images)} images)")
                print("  New files will be added with sequential numbering after existing ones.")

        print()
        print("-" * 60)
        for i, d in enumerate(input_dirs, 1):
            print(f"{YELLOW}  1.{i} Input: {d}{RESET}" if len(input_dirs) > 1 else f"{YELLOW}  1. Input:  {d}{RESET}")
        print(f"{YELLOW}  2. Output: {output_dir}{RESET}")
        if include_labels:
            label_display = LABEL_FORMATS[label_format]['name']
            if config.get("convert_to_yolo"):
                label_display += " -> YOLO TXT (auto-converted)"
            print(f"{YELLOW}     Labels: {label_display}{RESET}")
        print(f"{YELLOW}     Mode:   {'Copy' if copy_files else 'Move'}{RESET}")
        print("-" * 60)
        prompt_options = "1=change input, 2=change output, 3=add input, Enter=proceed, q=cancel"
        choice = input(f"{GREEN}[{prompt_options}]: {RESET}").strip().lower()
        if choice == '':
            break
        elif choice == 'q':
            print("Consolidation cancelled.")
            return False
        elif choice == '1':
            if len(input_dirs) > 1:
                print("Current input directories:")
                for i, d in enumerate(input_dirs, 1):
                    print(f"  {i}. {d}")
                idx_input = input(f"{GREEN}Enter number to replace (or 'r' + number to remove, e.g. r2): {RESET}").strip()
                if idx_input.startswith('r') and idx_input[1:].isdigit():
                    rm_idx = int(idx_input[1:]) - 1
                    if 0 <= rm_idx < len(input_dirs) and len(input_dirs) > 1:
                        removed = input_dirs.pop(rm_idx)
                        print(f"Removed: {removed}")
                    else:
                        print(f"{RED}Invalid index.{RESET}")
                elif idx_input.isdigit():
                    replace_idx = int(idx_input) - 1
                    if 0 <= replace_idx < len(input_dirs):
                        new_input = input(f"{GREEN}Enter new input directory [{input_dirs[replace_idx]}]: {RESET}").strip()
                        if new_input:
                            if os.path.exists(new_input):
                                input_dirs[replace_idx] = new_input
                            else:
                                print(f"{RED}Error: Directory '{new_input}' does not exist.{RESET}")
                    else:
                        print(f"{RED}Invalid index.{RESET}")
            else:
                new_input = input(f"{GREEN}Enter new input directory [{input_dirs[0]}]: {RESET}").strip()
                if new_input:
                    if os.path.exists(new_input):
                        input_dirs[0] = new_input
                    else:
                        print(f"{RED}Error: Directory '{new_input}' does not exist.{RESET}")
        elif choice == '2':
            new_output = input(f"{GREEN}Enter new output directory [{output_dir}]: {RESET}").strip()
            if new_output:
                output_dir = new_output
        elif choice == '3':
            new_dir = input(f"{GREEN}Enter additional input directory: {RESET}").strip()
            if new_dir:
                if os.path.exists(new_dir):
                    input_dirs.append(new_dir)
                    print(f"Added: {new_dir}")
                else:
                    print(f"{RED}Error: Directory '{new_dir}' does not exist.{RESET}")

    try:
        for source_dir in input_dirs:
            print(f"\nConsolidating: {source_dir}")
            consolidator = ImageConsolidator(
                source_dir=source_dir,
                output_dir=output_dir,
                extensions=extensions,
                copy_files=copy_files,
                include_labels=include_labels,
                separate_folders=separate_folders,
                label_format=label_format
            )
            consolidator.consolidate()

        # Store for review step and YOLO conversion
        config["review_input_dir"] = output_dir
        config["label_format"] = label_format
        config["consolidate_input_dirs"] = input_dirs  # For YOLO class file lookup

        # Check if YOLO conversion is requested
        if config.get("convert_to_yolo"):
            yolo_success = _convert_to_yolo_format(
                output_dir=output_dir,
                train_ratio=config.get("yolo_train_ratio", 0.8),
                classes_file=config.get("yolo_classes_file")
            )
            if not yolo_success:
                return False

    except Exception as e:
        print(f"{RED}Error during consolidation: {e}{RESET}")
        return False

    return True


# ============================================================================
# YOLO FORMAT CONVERSION
# ============================================================================

def _extract_annotations_from_json(data: dict) -> list:
    """Extract annotation list from LabelMe/X-AnyLabeling JSON format."""
    boxes = []
    if 'shapes' not in data:
        return boxes

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
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
            else:
                continue

            boxes.append({
                'box': {'x': min(x1, x2), 'y': min(y1, y2),
                        'w': abs(x2 - x1), 'h': abs(y2 - y1)},
                'category_name': shape.get('label', 'unknown'),
                'img_width': img_width,
                'img_height': img_height
            })
    return boxes


def _convert_box_to_yolo(box: dict, img_width: int, img_height: int) -> tuple:
    """Convert box coordinates to YOLO format (normalized x_center, y_center, width, height)."""
    x = box.get('x', 0)
    y = box.get('y', 0)
    w = box.get('w', box.get('width', 0))
    h = box.get('h', box.get('height', 0))

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


def _build_class_mapping(label_dir: Path) -> tuple:
    """
    Build class mapping from JSON labels.
    Returns: (class_to_id dict, class_names list)
    """
    print("Scanning JSON files to extract class names...")
    class_names_set = set()

    for json_file in label_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = _extract_annotations_from_json(data)
            for ann in annotations:
                if 'category_name' in ann:
                    class_names_set.add(ann['category_name'])
        except Exception as e:
            print(f"{RED}Warning: Could not parse {json_file}: {e}{RESET}")

    class_names = sorted(list(class_names_set))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    return class_to_id, class_names


def _process_json_to_yolo(json_file: Path, image_dir: Path, output_label_dir: Path,
                          class_to_id: dict, output_image_dir: Path) -> int:
    """Process a JSON file and convert to YOLO format."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"{RED}Warning: Could not read {json_file}: {e}{RESET}")
        return 0

    # Get image name
    image_name = data.get('imagePath', json_file.stem)
    image_stem = Path(image_name).stem

    # Find corresponding image
    image_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.PNG', '.JPG', '.JPEG']:
        candidate = image_dir / f"{image_stem}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if not image_path:
        print(f"{RED}Warning: No image found for {json_file}{RESET}")
        return 0

    # Get image dimensions
    img_width = data.get('imageWidth', 0)
    img_height = data.get('imageHeight', 0)

    if img_width == 0 or img_height == 0:
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size

    # Extract annotations
    annotations = _extract_annotations_from_json(data)

    if not annotations:
        return 0

    # Write YOLO format labels
    output_label_path = output_label_dir / f"{image_path.stem}.txt"

    lines = []
    for ann in annotations:
        box = ann.get('box', ann)
        category = ann.get('category_name', 'unknown')

        if category not in class_to_id:
            print(f"{RED}Warning: Unknown class '{category}' in {json_file}{RESET}")
            continue

        class_id = class_to_id[category]

        try:
            x_center, y_center, w, h = _convert_box_to_yolo(box, img_width, img_height)
            if w > 0 and h > 0:
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        except Exception as e:
            print(f"{RED}Warning: Could not convert box in {json_file}: {e}{RESET}")

    if lines:
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(lines))

        # Copy image
        shutil.copy2(image_path, output_image_dir / image_path.name)
        return 1

    return 0


def _convert_to_yolo_format(output_dir: str, train_ratio: float = 0.8,
                            classes_file: str = None) -> bool:
    """
    Convert consolidated dataset from JSON to YOLO format.

    Args:
        output_dir: Directory containing Image/ and Label/ folders
        train_ratio: Ratio of training data (rest goes to validation)
        classes_file: Optional path to classes.txt (not used currently)

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)

    # Detect directory structure
    image_dir = output_path / 'Image'
    label_dir = output_path / 'Label'

    if not image_dir.exists() or not label_dir.exists():
        print(f"{RED}Error: Expected 'Image' and 'Label' folders in {output_dir}{RESET}")
        return False

    # Build class mapping
    class_to_id, class_names = _build_class_mapping(label_dir)
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

    if not json_files:
        print(f"{RED}No JSON files found in {label_dir}{RESET}")
        return False

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
        count = _process_json_to_yolo(json_file, image_dir, train_labels,
                                      class_to_id, train_images)
        train_count += count

    # Process validation data
    print("Processing validation data...")
    val_count = 0
    for json_file in val_jsons:
        count = _process_json_to_yolo(json_file, image_dir, val_labels,
                                      class_to_id, val_images)
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
# Auto-generated by consolidate.py

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

    # Remove intermediate Image/ and Label/ folders
    print("\nRemoving intermediate Image/ and Label/ folders...")
    shutil.rmtree(image_dir)
    shutil.rmtree(label_dir)
    print(f"  Removed: {image_dir}")
    print(f"  Removed: {label_dir}")

    print(f"\nYOLO dataset ready for training.")
    print(f"Use with: python train_yolo.py --data {yaml_path}")

    return True
