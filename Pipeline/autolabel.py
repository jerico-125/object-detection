#!/usr/bin/env python3
"""
Step: YOLO Auto-Labeling

Runs a trained YOLO model on images to generate labels automatically.
Images without any detections are deleted.
Saves labels in JSON (LabelMe/X-AnyLabeling) format for review in X-AnyLabeling.
"""

import os
import json
import shutil
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Default directory for YOLO version runs
DEFAULT_RUNS_DIR = "/home/aidall/AI_Hub/runs/detect/runs"


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


def prompt_with_default_value(prompt_text: str, default_value) -> str:
    """Prompt user for input, showing default value. Returns default if empty input."""
    default_str = str(default_value) if default_value is not None else ""
    if default_str:
        user_input = input(f"{GREEN}{prompt_text} [default: {default_str}]: {RESET}").strip()
    else:
        user_input = input(f"{GREEN}{prompt_text}: {RESET}").strip()
    return user_input if user_input else default_str


# ============================================================================
# YOLO LABEL GENERATION
# ============================================================================

def create_labelme_json(image_path: Path, image_shape: tuple, detections: list) -> dict:
    """
    Create a LabelMe/X-AnyLabeling compatible JSON label from YOLO detections.

    Args:
        image_path: Path to the image file
        image_shape: (height, width, channels) of the image
        detections: List of dicts with keys: label, x1, y1, x2, y2, confidence

    Returns:
        Dictionary in LabelMe JSON format
    """
    height, width = image_shape[:2]

    shapes = []
    for det in detections:
        x1, y1 = float(det["x1"]), float(det["y1"])
        x2, y2 = float(det["x2"]), float(det["y2"])
        shapes.append({
            "label": det["label"],
            "points": [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ],
            "group_id": None,
            "description": f"conf:{det['confidence']:.2f}",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {}
        })

    label_data = {
        "version": "0.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    return label_data


def run_yolo_inference(
    model_path: str,
    input_dir: str,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "",
    delete_unlabeled: bool = True,
    deleted_dir: str = "./deleted/unlabeled",
    output_dir: str = "",
) -> dict:
    """
    Run YOLO model on all images in a directory, save JSON labels,
    and optionally move images with no detections to deleted directory.

    Labeled images and their JSON labels are copied to output_dir if provided.

    Args:
        model_path: Path to YOLO model weights (.pt or .onnx)
        input_dir: Directory containing images to label
        confidence: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        imgsz: Input image size for inference
        device: CUDA device or 'cpu'
        delete_unlabeled: If True, move images without detections to deleted_dir
        deleted_dir: Directory to move unlabeled images to
        output_dir: Directory to copy labeled images and labels into (if empty, labels saved in-place)

    Returns:
        Dictionary with statistics
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    input_path = Path(input_dir)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = sorted([
        f for f in input_path.rglob('*')
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return {'total': 0, 'labeled': 0, 'unlabeled': 0, 'deleted': 0, 'total_detections': 0}

    # Create output directory for labeled images if specified
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(image_files)} images")
    if output_path:
        print(f"Copy labeled to: {output_path}")
    print(f"Move unlabeled to: {deleted_dir}")
    Path(deleted_dir).mkdir(parents=True, exist_ok=True)
    print("-" * 50)

    stats = {
        'total': len(image_files),
        'labeled': 0,
        'unlabeled': 0,
        'deleted': 0,
        'total_detections': 0,
        'class_counts': {},
    }

    # ONNX models are exported with fixed batch size (usually 1),
    # so process one image at a time. For .pt models, use larger batches.
    is_onnx = str(model_path).lower().endswith('.onnx')
    BATCH_SIZE = 1 if is_onnx else 500
    unlabeled_files = []
    pbar = tqdm(total=len(image_files), desc="Auto-labeling", unit="img")

    for batch_start in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[batch_start:batch_start + BATCH_SIZE]
        source = str(batch[0]) if BATCH_SIZE == 1 else [str(f) for f in batch]

        predict_args = {
            'source': source,
            'conf': confidence,
            'iou': iou_threshold,
            'imgsz': imgsz,
            'save': False,
            'verbose': False,
            'stream': True,
        }
        if device:
            predict_args['device'] = device

        results = model.predict(**predict_args)

        for result in results:
            image_path = Path(result.path)
            boxes = result.boxes

            if len(boxes) == 0:
                stats['unlabeled'] += 1
                unlabeled_files.append(image_path)
                pbar.update(1)
                continue

            # Extract detections
            detections = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    'label': cls_name,
                    'confidence': conf,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                })

                # Track class counts
                stats['class_counts'][cls_name] = stats['class_counts'].get(cls_name, 0) + 1

            stats['labeled'] += 1
            stats['total_detections'] += len(detections)

            # Save JSON label file alongside the image
            label_data = create_labelme_json(
                image_path=image_path,
                image_shape=result.orig_shape,
                detections=detections
            )

            label_path = image_path.with_suffix('.json')
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=2, ensure_ascii=False)

            # Copy labeled image and label to output directory
            if output_path:
                shutil.copy2(str(image_path), str(output_path / image_path.name))
                shutil.copy2(str(label_path), str(output_path / label_path.name))

            pbar.update(1)

    pbar.close()

    # Move unlabeled images to deleted directory
    if delete_unlabeled and unlabeled_files:
        print(f"\nMoving {len(unlabeled_files)} images with no detections to: {deleted_dir}")
        for img_path in unlabeled_files:
            shutil.move(str(img_path), str(Path(deleted_dir) / img_path.name))
            stats['deleted'] += 1

    return stats


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def yolo_autolabel(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Run YOLO model on images to auto-generate labels and remove unlabeled images."""

    # Check for ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"{RED}Error: ultralytics package not found.{RESET}")
        print("Install it with: pip install ultralytics")
        return False

    # Get video name from previous steps
    video_name = config.get("video_name", "")
    if not video_name and config.get("video_path"):
        video_name = Path(config.get("video_path")).stem

    # Determine input directory
    if from_previous_step and config.get("extracted_frames_dir"):
        input_dir = config.get("extracted_frames_dir")
    else:
        base_dir = config.get("autolabel_input_dir", "./extracted_frames")
        input_dir = prompt_with_directory_options(
            "Enter the directory containing images to auto-label",
            base_dir,
            video_name
        )
        if not input_dir:
            print(f"{RED}No directory provided.{RESET}")
            return False

    if not os.path.exists(input_dir):
        print(f"{RED}Error: Input directory '{input_dir}' does not exist.{RESET}")
        return False

    # Get model path — interactive selection from YOLO_v* versions
    from model_utils import select_yolo_model
    runs_dir = config.get("yolo_runs_dir", DEFAULT_RUNS_DIR)
    model_path = select_yolo_model(runs_dir=runs_dir)
    if not model_path or not os.path.exists(model_path):
        print(f"{RED}Error: Model file not found: {model_path}{RESET}")
        return False

    # Get inference parameters
    confidence = config.get("autolabel_confidence", 0.25)
    iou_threshold = config.get("autolabel_iou", 0.45)
    imgsz = config.get("autolabel_imgsz", 640)
    device = config.get("autolabel_device", "")
    delete_unlabeled = config.get("autolabel_delete_unlabeled", True)
    deleted_dir = config.get("autolabel_deleted_dir", "./deleted/unlabeled")

    # Output directory for labeled images and labels
    output_dir = config.get("autolabel_output_dir", "./autolabeled")
    if video_name:
        output_path = Path(output_dir) / video_name
    else:
        output_path = Path(output_dir)
    output_dir = str(output_path)

    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_path}")
    print(f"Confidence: {confidence}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Image size: {imgsz}")
    print(f"Delete unlabeled: {delete_unlabeled}")
    print()

    try:
        stats = run_yolo_inference(
            model_path=model_path,
            input_dir=input_dir,
            confidence=confidence,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            device=device,
            delete_unlabeled=delete_unlabeled,
            deleted_dir=deleted_dir,
            output_dir=output_dir,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("AUTO-LABELING SUMMARY")
        print("=" * 60)
        print(f"Total images processed: {stats['total']}")
        print(f"Images with detections: {stats['labeled']}")
        print(f"Images without detections: {stats['unlabeled']}")
        print(f"Images removed: {stats['deleted']}")
        print(f"Total detections: {stats['total_detections']}")

        if stats['class_counts']:
            print("\nDetections per class:")
            for cls_name, count in sorted(stats['class_counts'].items()):
                print(f"  {cls_name}: {count}")

        print("=" * 60)

        # Update config for next step — point to output directory with labeled images
        config["labeling_input_dir"] = output_dir
        config["autolabel_output_result_dir"] = output_dir

        return True

    except Exception as e:
        print(f"{RED}Error during auto-labeling: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False
