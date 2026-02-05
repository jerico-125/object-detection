#!/usr/bin/env python3
"""
YOLOv8 Training Script

Usage:
    python train_yolo.py --data dataset.yaml --epochs 100
    python train_yolo.py --data dataset.yaml --model yolov8m.pt --epochs 200 --batch 8
    python train_yolo.py --data dataset.yaml --venv /path/to/yolo_env  # Auto-activate venv

Requirements:
    pip install ultralytics
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Note: YOLO_VERBOSE=false was removed because it suppresses training progress output

# Default virtual environment path
DEFAULT_VENV_PATH = "/home/aidall/AI_Hub/yolo_env"


def activate_venv(venv_path: str) -> bool:
    """
    Activate a virtual environment by modifying sys.path and environment variables.

    Args:
        venv_path: Path to the virtual environment directory

    Returns:
        True if activation successful, False otherwise
    """
    venv_path = Path(venv_path).resolve()

    if not venv_path.exists():
        print(f"Virtual environment not found at: {venv_path}")
        print(f"Create it with: python3 -m venv {venv_path}")
        return False

    # Determine the correct bin/Scripts directory
    if sys.platform == "win32":
        bin_dir = venv_path / "Scripts"
        python_exe = bin_dir / "python.exe"
    else:
        bin_dir = venv_path / "bin"
        python_exe = bin_dir / "python"

    if not python_exe.exists():
        print(f"Python executable not found in venv: {python_exe}")
        return False

    # Check if we're already running from this venv
    current_prefix = getattr(sys, 'prefix', '')
    if str(venv_path) == current_prefix:
        print(f"Already running in virtual environment: {venv_path}")
        return True

    # If not in the correct venv, re-execute the script with the venv's Python
    print(f"Activating virtual environment: {venv_path}")

    # Re-run this script using the venv's Python interpreter
    new_env = os.environ.copy()
    new_env["VIRTUAL_ENV"] = str(venv_path)
    new_env["PATH"] = f"{bin_dir}{os.pathsep}{new_env.get('PATH', '')}"

    # Remove any existing PYTHONHOME which can interfere
    new_env.pop("PYTHONHOME", None)

    # Re-execute with the venv Python
    result = subprocess.run(
        [str(python_exe)] + sys.argv,
        env=new_env,
        cwd=os.getcwd()
    )

    # Exit with the same code as the subprocess
    sys.exit(result.returncode)


def setup_venv(venv_path: str) -> bool:
    """
    Create virtual environment and install dependencies if it doesn't exist.

    Args:
        venv_path: Path to create the virtual environment

    Returns:
        True if setup successful, False otherwise
    """
    venv_path = Path(venv_path).resolve()

    if venv_path.exists():
        return True

    print(f"Creating virtual environment at: {venv_path}")

    try:
        # Create venv
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True
        )

        # Determine pip path
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"

        # Upgrade pip
        print("Upgrading pip...")
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            check=True
        )

        # Install dependencies
        print("Installing ultralytics and dependencies...")
        subprocess.run(
            [str(pip_path), "install", "ultralytics", "pillow"],
            check=True
        )

        print(f"Virtual environment created successfully at: {venv_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False


def is_in_venv() -> bool:
    """Check if currently running inside a virtual environment."""
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )


def check_dependencies():
    """Check if required packages are installed."""
    try:
        from ultralytics import YOLO
        print("Ultralytics YOLO is installed.")
        return True
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Install it with: pip install ultralytics")
        return False


def train_yolo(
    data: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "",
    workers: int = 8,
    project: str = "./YOLO_result",
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
    optimizer: str = "auto",
    lr0: float = 0.01,
    patience: int = 50,
    save_period: int = -1,
    cache: bool = False,
    amp: bool = True,
    freeze: int = None,
    augment: bool = True,
    visualize_samples: int = 20,
    vis_conf: float = 0.33,
):
    """
    Train YOLOv8 model.

    Args:
        data: Path to dataset YAML file
        model: Model to use (yolov8n/s/m/l/x.pt or path to custom weights)
        epochs: Number of training epochs
        batch: Batch size (-1 for auto-batch)
        imgsz: Input image size
        device: CUDA device (e.g., '0' or '0,1,2,3' or 'cpu')
        workers: Number of dataloader workers
        project: Project directory for saving results
        name: Experiment name
        resume: Resume training from last checkpoint
        pretrained: Use pretrained weights
        optimizer: Optimizer (SGD, Adam, AdamW, auto)
        lr0: Initial learning rate
        patience: Early stopping patience (0 to disable)
        save_period: Save checkpoint every N epochs (-1 to disable)
        cache: Cache images for faster training
        amp: Use Automatic Mixed Precision
        freeze: Freeze first N layers
        augment: Use data augmentation
        visualize_samples: Number of sample images to visualize after training (0 to skip)
        vis_conf: Confidence threshold for visualization
    """
    from ultralytics import YOLO

    # Validate data file exists
    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset config not found: {data}")

    # Generate experiment name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model).stem
        name = f"{model_name}_{timestamp}"

    print("=" * 60)
    print("YOLOv8 Training Configuration")
    print("=" * 60)
    print(f"  Model:      {model}")
    print(f"  Dataset:    {data}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Device:     {device if device else 'auto'}")
    print(f"  Project:    {project}")
    print(f"  Name:       {name}")
    print("=" * 60)

    # Load model
    if resume:
        # Resume from last checkpoint
        model_path = Path(project) / name / "weights" / "last.pt"
        if model_path.exists():
            yolo = YOLO(str(model_path))
            print(f"Resuming from: {model_path}")
        else:
            print(f"No checkpoint found at {model_path}, starting fresh")
            yolo = YOLO(model)
    else:
        yolo = YOLO(model)

    # Training arguments
    train_args = {
        "data": data,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "project": project,
        "name": name,
        "exist_ok": True,
        "pretrained": pretrained,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
        "save_period": save_period,
        "cache": cache,
        "amp": amp,
        "verbose": True,
        "plots": True,
    }

    # Add device if specified
    if device:
        train_args["device"] = device

    # Add freeze if specified
    if freeze is not None:
        train_args["freeze"] = freeze

    # Augmentation settings
    if augment:
        train_args.update({
            "hsv_h": 0.015,  # HSV-Hue augmentation
            "hsv_s": 0.7,    # HSV-Saturation augmentation
            "hsv_v": 0.4,    # HSV-Value augmentation
            "degrees": 0.0,  # Rotation
            "translate": 0.1,  # Translation
            "scale": 0.5,    # Scale
            "shear": 0.0,    # Shear
            "perspective": 0.0,  # Perspective
            "flipud": 0.0,   # Flip up-down
            "fliplr": 0.5,   # Flip left-right
            "mosaic": 1.0,   # Mosaic augmentation
            "mixup": 0.0,    # Mixup augmentation
        })

    # Start training
    print("\nStarting training...")
    results = yolo.train(**train_args)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {results.save_dir}")

    # Prompt user for sample visualization
    print("\n" + "-" * 60)
    run_vis = input("Run sample visualization on validation images? [Y/n]: ").strip().lower()

    if run_vis in ('', 'y', 'yes'):
        samples_input = input(f"Number of samples [{visualize_samples}]: ").strip()
        conf_input = input(f"Confidence threshold [{vis_conf}]: ").strip()

        num_samples = int(samples_input) if samples_input else visualize_samples
        conf_threshold = float(conf_input) if conf_input else vis_conf

        import random
        import yaml
        import cv2

        print(f"\nVisualizing {num_samples} sample predictions (conf={conf_threshold})...")

        # Load best weights
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            best_weights = Path(results.save_dir) / "weights" / "last.pt"

        best_model = YOLO(str(best_weights))

        # Get validation image directory from dataset.yaml
        with open(data, 'r') as f:
            dataset_cfg = yaml.safe_load(f)

        dataset_root = Path(dataset_cfg.get('path', ''))
        val_dir = dataset_root / dataset_cfg.get('val', 'images/val')

        if not val_dir.exists():
            print(f"Warning: Validation directory not found: {val_dir}")
            return results

        # Collect validation images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        val_images = [
            f for f in val_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not val_images:
            print("Warning: No validation images found")
            return results

        # Sample randomly
        num_samples = min(num_samples, len(val_images))
        sampled = random.sample(val_images, num_samples)

        # Create output directory
        vis_dir = Path(results.save_dir) / "sample_predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, img_file in enumerate(sampled):
            pred_results = best_model.predict(
                source=str(img_file),
                conf=conf_threshold,
                imgsz=imgsz,
                verbose=False
            )

            for result in pred_results:
                annotated = result.plot(labels=True, conf=True, line_width=2)
                out_path = vis_dir / f"sample_{i+1}_{img_file.stem}.jpg"
                cv2.imwrite(str(out_path), annotated)

                num_det = len(result.boxes)
                det_summary = []
                for box in result.boxes:
                    cls_name = best_model.names[int(box.cls[0])]
                    cls_conf = float(box.conf[0])
                    det_summary.append(f"{cls_name}({cls_conf:.2f})")

                print(f"  [{i+1}/{num_samples}] {img_file.name}: "
                      f"{num_det} detections - {', '.join(det_summary) if det_summary else 'none'}")

        print(f"\nSample predictions saved to: {vis_dir}")
    else:
        print("Skipping visualization.")

    # Export to ONNX
    print("\n" + "-" * 60)
    run_export = input("Export model to ONNX format? [Y/n]: ").strip().lower()

    if run_export in ('', 'y', 'yes'):
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            best_weights = Path(results.save_dir) / "weights" / "last.pt"

        print(f"Exporting {best_weights} to ONNX...")
        export_model = YOLO(str(best_weights))
        export_model.export(format='onnx')
        onnx_path = best_weights.with_suffix('.onnx')
        print(f"ONNX model saved to: {onnx_path}")

        # Generate X-AnyLabeling config
        import yaml
        with open(data, 'r') as f:
            dataset_cfg = yaml.safe_load(f)

        class_names = dataset_cfg.get('names', {})
        if isinstance(class_names, dict):
            class_list = [class_names[k] for k in sorted(class_names.keys())]
        else:
            class_list = list(class_names)

        config_path = Path(results.save_dir) / "weights" / "x_anylabeling_config.yaml"
        config_content = {
            'type': 'yolov8',
            'name': name,
            'display_name': name,
            'model_path': str(onnx_path),
            'input_width': imgsz,
            'input_height': imgsz,
            'score_threshold': 0.25,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'iou_threshold': 0.45,
            'classes': class_list,
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False, sort_keys=False)
        print(f"X-AnyLabeling config saved to: {config_path}")
    else:
        print("Skipping ONNX export.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )

    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        help="Model to use: yolov8n/s/m/l/x.pt or path to weights"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Batch size (-1 for auto-batch)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="CUDA device (e.g., '0' or '0,1' or 'cpu')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers"
    )

    # Output arguments
    parser.add_argument(
        "--project",
        type=str,
        default="./YOLO_result",
        help="Project directory for results (default: ./YOLO_result)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name"
    )

    # Training options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use pretrained weights"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer to use"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (0 to disable)"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze first N layers"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )

    # Visualization arguments
    parser.add_argument(
        "--vis-samples",
        type=int,
        default=20,
        help="Default number of sample images for visualization prompt"
    )
    parser.add_argument(
        "--vis-conf",
        type=float,
        default=0.33,
        help="Default confidence threshold for visualization prompt"
    )

    # Virtual environment arguments
    parser.add_argument(
        "--venv",
        type=str,
        default=DEFAULT_VENV_PATH,
        help=f"Path to virtual environment (default: {DEFAULT_VENV_PATH})"
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Skip virtual environment activation"
    )
    parser.add_argument(
        "--setup-venv",
        action="store_true",
        help="Create virtual environment and install dependencies if needed"
    )

    args = parser.parse_args()

    # Handle virtual environment
    if not args.no_venv and not is_in_venv():
        if args.setup_venv:
            if not setup_venv(args.venv):
                print("Failed to setup virtual environment")
                return

        if Path(args.venv).exists():
            # This will re-execute the script in the venv and exit
            activate_venv(args.venv)
        else:
            print(f"Warning: Virtual environment not found at {args.venv}")
            print("Run with --setup-venv to create it, or --no-venv to skip")
            return

    # Check dependencies
    if not check_dependencies():
        return

    # Run training
    train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=not args.no_pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        amp=not args.no_amp,
        freeze=args.freeze,
        augment=not args.no_augment,
        visualize_samples=args.vis_samples,
        vis_conf=args.vis_conf,
    )


if __name__ == "__main__":
    main()
