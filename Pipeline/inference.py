"""
Standalone YOLO Video Inference

Two modes:
  1. Real-time mode (--show): Opens a window with live bounding boxes, FPS counter,
     and detection count. Press 'q' to quit.
  2. Batch mode (default): Processes the video and saves the annotated output.
     No display window.

GPU acceleration:
  - YOLO inference uses GPU automatically if CUDA is available.
  - Use --half for FP16 half-precision inference (faster on GPUs with Tensor Cores).
  - Use --device to select a specific GPU (e.g. --device 0).

Usage:
    python inference.py --source video.mp4 --model best.pt              # batch (save)
    python inference.py --source video.mp4 --model best.pt --show       # real-time display
    python inference.py --source video.mp4 --model best.pt --show --save  # both
    python inference.py --source 0 --model best.pt --show               # webcam
    python inference.py --source video.mp4 --model best.pt --half       # FP16 acceleration
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Default directory for YOLO version runs
DEFAULT_RUNS_DIR = "/home/aidall/Object_Detection/runs/detect/runs"


def print_device_info(device):
    """Print GPU/CPU device information."""
    if torch.cuda.is_available():
        if device and device != "cpu":
            dev_idx = int(device.split(",")[0])
        else:
            dev_idx = 0
        gpu_name = torch.cuda.get_device_name(dev_idx)
        gpu_mem = torch.cuda.get_device_properties(dev_idx).total_mem / 1024**3
        print(f"Device: GPU — {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("Device: CPU (no CUDA GPU detected)")


def run_realtime(model, conf, iou, imgsz, device, classes, save, half,
                 cap, width, height, video_fps, total_frames):
    """Real-time mode: display video with live detections in an OpenCV window."""
    print("Mode: Real-time display")
    print("Press 'q' to quit\n")

    writer = None
    if save:
        output_path = f"output_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        print(f"Saving output to: {output_path}")

    frame_count = 0
    fps_smoothed = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        results = model.predict(
            frame, conf=conf, iou=iou, imgsz=imgsz,
            device=device if device else None, classes=classes,
            half=half, verbose=False,
        )

        result = results[0]
        annotated = result.plot()

        # Smoothed FPS
        t_elapsed = time.time() - t_start
        fps = 1.0 / t_elapsed if t_elapsed > 0 else 0
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps if fps_smoothed > 0 else fps

        # Overlay FPS and detection count
        det_count = len(result.boxes)
        cv2.putText(annotated, f"FPS: {fps_smoothed:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"Detections: {det_count}", (10, 65),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("YOLO Inference", annotated)

        if writer:
            writer.write(annotated)

        frame_count += 1

        # Terminal progress
        if total_frames > 0:
            pct = frame_count / total_frames * 100
            print(f"\rFrame {frame_count}/{total_frames} ({pct:.0f}%) | "
                  f"FPS: {fps_smoothed:.1f} | Detections: {det_count}", end="", flush=True)
        else:
            print(f"\rFrame {frame_count} | FPS: {fps_smoothed:.1f} | "
                  f"Detections: {det_count}", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped by user.")
            break

    print(f"\nDone. Processed {frame_count} frames.")

    if writer:
        writer.release()
        print(f"Output saved to: {output_path}")
    cv2.destroyAllWindows()


def run_batch(model, conf, iou, imgsz, device, classes, half,
              cap, width, height, video_fps, total_frames):
    """Batch mode: process video and save annotated output without display."""
    output_path = f"output_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    print(f"Mode: Batch processing (no display)")
    print(f"Saving output to: {output_path}\n")

    frame_count = 0
    fps_smoothed = 0.0
    t_total_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        results = model.predict(
            frame, conf=conf, iou=iou, imgsz=imgsz,
            device=device if device else None, classes=classes,
            half=half, verbose=False,
        )

        result = results[0]
        annotated = result.plot()
        writer.write(annotated)

        # Per-frame FPS
        t_elapsed = time.time() - t_start
        fps = 1.0 / t_elapsed if t_elapsed > 0 else 0
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps if fps_smoothed > 0 else fps

        frame_count += 1
        det_count = len(result.boxes)

        if total_frames > 0:
            pct = frame_count / total_frames * 100
            print(f"\rFrame {frame_count}/{total_frames} ({pct:.0f}%) | "
                  f"FPS: {fps_smoothed:.1f} | Detections: {det_count}", end="", flush=True)
        else:
            print(f"\rFrame {frame_count} | FPS: {fps_smoothed:.1f} | "
                  f"Detections: {det_count}", end="", flush=True)

    elapsed = time.time() - t_total_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"\nDone. Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} avg FPS).")
    print(f"Output saved to: {output_path}")

    writer.release()


def run_inference(source, model_path, conf=0.25, iou=0.45, imgsz=640,
                  show=False, save=False, device="", classes=None, half=False):
    """Run YOLO inference on a video source.

    Args:
        source: Video file path, camera index (0, 1, ...), or RTSP/HTTP stream URL.
        model_path: Path to YOLO model (.pt or .onnx).
        conf: Confidence threshold (0-1).
        iou: IoU threshold for NMS (0-1).
        imgsz: Input image size.
        show: Real-time display mode with OpenCV window.
        save: Save annotated video to disk (always True in batch mode).
        device: CUDA device (e.g. '0', '0,1', 'cpu'). Empty string = auto.
        classes: List of class indices to filter (e.g. [0, 2, 5]).
        half: Use FP16 half-precision inference.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"\033[91mError: Could not open video source: {source}\033[0m")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source: {source}")
    print(f"Resolution: {width}x{height} | Video FPS: {video_fps:.1f}", end="")
    if total_frames > 0:
        duration = total_frames / video_fps
        print(f" | Total frames: {total_frames} ({duration:.1f}s)")
    else:
        print()
    print(f"Model: {model_path}")
    print_device_info(device)
    print(f"Confidence: {conf} | IoU: {iou} | Image size: {imgsz}"
          f"{' | FP16: on' if half else ''}")

    if show:
        run_realtime(model, conf, iou, imgsz, device, classes, save, half,
                     cap, width, height, video_fps, total_frames)
    else:
        run_batch(model, conf, iou, imgsz, device, classes, half,
                  cap, width, height, video_fps, total_frames)

    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on a video with real-time or batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Batch (default):  Processes video and saves annotated output. No display window.
  Real-time (--show): Opens a window with live bounding boxes, FPS, and detection count.

GPU acceleration:
  YOLO uses GPU automatically if CUDA is available. Use --half for FP16 speed boost.

Examples:
  python inference.py --source video.mp4                              # uses latest YOLO_v*
  python inference.py --source video.mp4 --model best.pt              # specific model
  python inference.py --source video.mp4 --show                       # real-time
  python inference.py --source video.mp4 --show --save                # real-time + save
  python inference.py --source 0 --show                               # webcam
  python inference.py --source video.mp4 --half                       # FP16 faster
  python inference.py --source video.mp4 --classes 0 2 5
        """,
    )

    parser.add_argument("--source", required=True,
                        help="Video file path, camera index (0, 1, ...), or stream URL")
    parser.add_argument("--model", default=None,
                        help="Path to YOLO model (.pt or .onnx). Default: latest YOLO_v* model")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--show", action="store_true",
                        help="Real-time mode: display video with live detections")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video (always on in batch mode)")
    parser.add_argument("--device", default="",
                        help="CUDA device (e.g. '0', '0,1', 'cpu'). Empty = auto")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Filter by class index (e.g. --classes 0 2 5)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 half-precision inference (faster on GPU)")

    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if model_path is None:
        from model_utils import select_yolo_model
        model_path = select_yolo_model(runs_dir=DEFAULT_RUNS_DIR)
        if not model_path:
            print("\033[91mError: No model specified.\033[0m")
            return

    # If source is a number, treat it as a camera index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run_inference(
        source=source,
        model_path=model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=args.show,
        save=args.save,
        device=args.device,
        classes=args.classes,
        half=args.half,
    )


if __name__ == "__main__":
    main()
