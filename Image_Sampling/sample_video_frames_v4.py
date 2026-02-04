#!/usr/bin/env python3
"""
Fast video frame sampling using ffmpeg streaming and histogram-based filtering.
Processes frames entirely in memory - no intermediate disk I/O.

Usage: python sample_video_frames_stream.py <video_path> [options]

Examples:
    python sample_video_frames_stream.py video.mp4 --output frames/
    python sample_video_frames_stream.py video.mp4 --threshold 0.85 --interval 10
    python sample_video_frames_stream.py video.mp4 --max-frames 500
"""

import os
import sys
import argparse
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to input video file

    Returns:
        Dictionary with video info (fps, width, height, duration, total_frames)
    """
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        info_parts = result.stdout.strip().split(',')

        # Find the frame rate field (contains '/')
        fps_idx = None
        for i, part in enumerate(info_parts):
            if '/' in part:
                fps_idx = i
                break

        if fps_idx is None:
            raise ValueError(f"Could not find frame rate in ffprobe output: {info_parts}")

        # Parse fps (e.g., "30/1" -> 30.0)
        fps_parts = info_parts[fps_idx].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])

        # Width and height are BEFORE fps
        width = int(info_parts[fps_idx - 2]) if fps_idx >= 2 else int(info_parts[0])
        height = int(info_parts[fps_idx - 1]) if fps_idx >= 1 else int(info_parts[1])

        # Duration is after fps
        duration = 0
        if len(info_parts) > fps_idx + 1 and info_parts[fps_idx + 1]:
            try:
                duration = float(info_parts[fps_idx + 1])
            except ValueError:
                pass

        # Total frames is after duration
        total_frames = 0
        if len(info_parts) > fps_idx + 2 and info_parts[fps_idx + 2]:
            try:
                total_frames = int(info_parts[fps_idx + 2])
            except ValueError:
                pass

        # If total_frames not available, estimate from duration and fps
        if total_frames == 0 and duration > 0:
            total_frames = int(duration * fps)

        return {
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'total_frames': total_frames
        }

    except subprocess.CalledProcessError as e:
        print(f"Error getting video info: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise
    except (ValueError, IndexError) as e:
        print(f"Error parsing ffprobe output: {e}", file=sys.stderr)
        print(f"Raw output: '{result.stdout.strip()}'", file=sys.stderr)
        print(f"Parsed parts: {info_parts}", file=sys.stderr)
        raise


def compute_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Compute normalized HSV histogram for a frame.

    Args:
        frame: BGR image frame
        bins: Number of bins per color channel

    Returns:
        Flattened normalized histogram array
    """
    # Downsample for faster processing
    small_frame = frame[::8, ::8]

    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

    # Compute histogram for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Normalize histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Concatenate all channels
    histogram = np.concatenate([hist_h, hist_s, hist_v])

    return histogram


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compare two histograms using correlation method.

    Args:
        hist1: First histogram
        hist2: Second histogram

    Returns:
        Similarity score (1.0 = identical, -1.0 = completely different)
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def stream_frames_ffmpeg(video_path: str,
                         frame_interval: int,
                         target_fps: float,
                         width: int,
                         height: int) -> Tuple[subprocess.Popen, int, int]:
    """
    Start ffmpeg process to stream frames to stdout.

    Args:
        video_path: Path to input video file
        frame_interval: Extract every N frames
        target_fps: Target frames per second after decimation
        width: Video width
        height: Video height

    Returns:
        Tuple of (Running subprocess.Popen object, output_width, output_height)
    """
    output_width = width
    output_height = height
    vf_filter = f'fps={target_fps}'

    # Build ffmpeg command to output raw BGR24 frames to stdout
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', vf_filter,  # Decimate frames and optionally scale
        '-f', 'rawvideo',  # Output raw frames
        '-pix_fmt', 'bgr24',  # OpenCV-compatible pixel format
        '-',  # Output to stdout
        '-hide_banner',
        '-loglevel', 'error'
    ]

    # Start ffmpeg process with pipe to stdout
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8  # Large buffer for performance
    )

    return process, output_width, output_height


def extract_frame_at_index(video_path: str, frame_index: int, output_file: str) -> bool:
    """
    Extract a single frame at full resolution using ffmpeg.

    Args:
        video_path: Path to input video file
        frame_index: Frame index to extract (0-based)
        output_file: Path to save the output frame

    Returns:
        True if successful, False otherwise
    """
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=eq(n\\,{frame_index})',
        '-vframes', '1',
        output_file,
        '-y',  # Overwrite output file
        '-hide_banner',
        '-loglevel', 'error'
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def sample_frames_streaming(video_path: str,
                           output_dir: str,
                           threshold: float = 0.85,
                           frame_interval: int = 10,
                           max_frames: Optional[int] = None,
                           bins: int = 32,
                           prefix: str = "frame",
                           profile: bool = False) -> List[str]:
    """
    Sample diverse frames from video using streaming (no intermediate disk I/O).

    Args:
        video_path: Path to input video file
        output_dir: Directory to save filtered frames
        threshold: Histogram similarity threshold (0-1, lower = less strict)
        frame_interval: Extract every N frames from video
        max_frames: Maximum number of frames to save (None = unlimited)
        bins: Number of bins per color channel for histogram
        prefix: Prefix for output frame filenames
        profile: Enable detailed performance profiling

    Returns:
        List of saved unique frame paths
    """
    import time
    start_time = time.time()

    # Profiling timers
    time_read = 0.0
    time_reshape = 0.0
    time_histogram = 0.0
    time_compare = 0.0
    time_save = 0.0

    # Get video info
    print("Getting video information...")
    video_info = get_video_info(video_path)

    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    duration = video_info['duration']
    total_frames = video_info['total_frames']

    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Frame interval: {frame_interval}")
    print(f"Similarity threshold: {threshold}")
    print("-" * 50)

    # Calculate target FPS and expected frame count
    target_fps = fps / frame_interval
    # Use duration-based calculation instead of total_frames for accuracy
    expected_frames = int(duration * target_fps) if duration > 0 else int(total_frames / frame_interval)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start ffmpeg streaming process
    print(f"Starting ffmpeg stream (target: {target_fps:.2f} fps, ~{expected_frames} frames)...")
    process, output_width, output_height = stream_frames_ffmpeg(
        video_path, frame_interval, target_fps, width, height
    )

    # Frame size in bytes (width * height * 3 channels)
    frame_size = output_width * output_height * 3

    saved_frames = []
    last_histogram = None
    saved_count = 0
    processed_count = 0

    try:
        with tqdm(total=expected_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                # Read one frame from stdout
                if profile:
                    t0 = time.time()
                raw_frame = process.stdout.read(frame_size)
                if profile:
                    time_read += time.time() - t0

                # Check if we've reached the end
                if len(raw_frame) != frame_size:
                    break

                # Convert raw bytes to numpy array
                if profile:
                    t0 = time.time()
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((output_height, output_width, 3))
                if profile:
                    time_reshape += time.time() - t0

                processed_count += 1

                # Compute histogram for current frame
                if profile:
                    t0 = time.time()
                current_hist = compute_histogram(frame, bins)
                if profile:
                    time_histogram += time.time() - t0

                # First frame is always kept
                is_unique = last_histogram is None
                similarity = 0.0

                if not is_unique:
                    # Compare with last saved frame
                    if profile:
                        t0 = time.time()
                    similarity = compare_histograms(last_histogram, current_hist)
                    if profile:
                        time_compare += time.time() - t0

                    # Keep frame if similarity is below threshold (i.e., different enough)
                    is_unique = similarity < threshold

                if is_unique:
                    # Save frame to output directory
                    if profile:
                        t0 = time.time()
                    output_filename = f"{prefix}_{saved_count:06d}.png"
                    output_file = output_path / output_filename
                    cv2.imwrite(str(output_file), frame)

                    if profile:
                        time_save += time.time() - t0

                    saved_frames.append(str(output_file))
                    last_histogram = current_hist
                    saved_count += 1

                # Update progress bar with real-time profiling
                if profile and processed_count % 10 == 0:  # Update every 10 frames
                    elapsed = time.time() - start_time
                    postfix = {
                        'saved': saved_count,
                        'read%': f'{time_read/elapsed*100:.0f}',
                        'hist%': f'{time_histogram/elapsed*100:.0f}',
                        'save%': f'{time_save/elapsed*100:.0f}'
                    }
                    pbar.set_postfix(postfix)
                elif not profile:
                    pbar.set_postfix({'saved': saved_count, 'sim': f'{similarity:.3f}'})

                # Check max frames limit
                if max_frames and saved_count >= max_frames:
                    print(f"\nReached maximum frame limit: {max_frames}")
                    break

                pbar.update(1)

    finally:
        # Cleanup ffmpeg process
        process.stdout.close()
        process.terminate()
        process.wait()

    total_time = time.time() - start_time

    # Calculate statistics
    extraction_ratio = processed_count / total_frames * 100 if total_frames > 0 else 0
    reduction_ratio = (1 - saved_count / processed_count) * 100 if processed_count > 0 else 0
    overall_ratio = (1 - saved_count / total_frames) * 100 if total_frames > 0 else 0

    print("-" * 50)
    print(f"Completed!")
    print(f"Processed: {processed_count} frames ({extraction_ratio:.1f}% of total)")
    print(f"Saved: {saved_count} unique frames ({reduction_ratio:.1f}% reduction)")
    print(f"Overall: {saved_count}/{total_frames} frames ({overall_ratio:.1f}% reduction)")
    print(f"Output directory: {output_dir}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Processing speed: {processed_count/total_time:.1f} frames/sec")

    return saved_frames


def main():
    parser = argparse.ArgumentParser(
        description='Fast video frame sampling using ffmpeg streaming and histogram filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract diverse frames with default settings
  python sample_video_frames_stream.py input.mp4 --output frames/

  # Adjust threshold (higher = more strict filtering)
  python sample_video_frames_stream.py input.mp4 --threshold 0.85 --output frames/

  # Extract every 5 frames and filter
  python sample_video_frames_stream.py input.mp4 --interval 5 --output frames/

  # Limit maximum number of frames
  python sample_video_frames_stream.py input.mp4 --max-frames 500 --output dataset/
        """
    )

    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output', '-o', default='./sampled_frames',
                       help='Output directory for frames (default: ./sampled_frames)')
    parser.add_argument('--threshold', '-t', type=float, default=0.85,
                       help='Histogram similarity threshold 0-1, higher = more strict (default: 0.85)')
    parser.add_argument('--fps', '-f', type=float, default=3.0,
                       help='Target frames per second to extract (default: 3.0)')
    parser.add_argument('--interval', '-i', type=int, default=None,
                       help='Extract every N frames (overrides --fps if specified)')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                       help='Maximum frames to save (default: unlimited)')
    parser.add_argument('--bins', '-b', type=int, default=32,
                       help='Histogram bins per channel (default: 32)')
    parser.add_argument('--prefix', '-p', default='frame',
                       help='Filename prefix (default: frame)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable real-time performance profiling in progress bar')

    args = parser.parse_args()

    # Check video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' does not exist.")
        sys.exit(1)

    # Validate threshold
    if not 0 <= args.threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        sys.exit(1)

    # Check if ffmpeg and ffprobe are available
    if not shutil.which('ffmpeg'):
        print("Error: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)

    if not shutil.which('ffprobe'):
        print("Error: ffprobe not found. Please install ffmpeg.")
        sys.exit(1)

    # Calculate frame_interval from target fps if not explicitly set
    if args.interval is None:
        # Get video info to determine original fps
        video_info = get_video_info(args.video)
        original_fps = video_info['fps']
        frame_interval = max(1, int(round(original_fps / args.fps)))
        print(f"Calculated frame interval: {frame_interval} (original fps: {original_fps:.2f}, target fps: {args.fps})")
    else:
        frame_interval = args.interval

    try:
        saved_frames = sample_frames_streaming(
            video_path=args.video,
            output_dir=args.output,
            threshold=args.threshold,
            frame_interval=frame_interval,
            max_frames=args.max_frames,
            bins=args.bins,
            prefix=args.prefix,
            profile=args.profile
        )

        if saved_frames:
            print(f"\nFirst frame: {saved_frames[0]}")
            print(f"Last frame: {saved_frames[-1]}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
