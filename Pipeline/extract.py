#!/usr/bin/env python3
"""
Step 1: Video Frame Extraction

Extract image frames from video files using ffmpeg streaming.
Supports both sequential deduplication and DBSCAN clustering-based deduplication.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


# ============================================================================
# VIDEO ANALYSIS UTILITIES
# ============================================================================

def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe with JSON output."""
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_read_packets',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if not data.get('streams'):
            raise ValueError(f"No video streams found in: {video_path}")

        stream = data['streams'][0]

        # Parse frame rate (format: "30/1" or "30000/1001")
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        duration = float(stream.get('duration', 0))
        total_frames = int(stream.get('nb_read_packets', 0))

        # Estimate total frames from duration if not available
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
        print(f"{RED}Error getting video info: {e}{RESET}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"{RED}Error parsing ffprobe JSON output: {e}{RESET}", file=sys.stderr)
        raise


# ============================================================================
# IMAGE QUALITY UTILITIES
# ============================================================================

def variance_of_laplacian(image: np.ndarray) -> float:
    """
    Compute the Laplacian of the image and return the focus measure.
    Higher values indicate sharper images, lower values indicate blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Global variables for multiprocessing worker (set by initializer)
_worker_blur_threshold = 100.0
_worker_bins = 32


def _init_worker(blur_threshold: float, bins: int):
    """Initialize worker process with shared parameters."""
    global _worker_blur_threshold, _worker_bins
    _worker_blur_threshold = blur_threshold
    _worker_bins = bins


def _process_single_image(img_path_str: str) -> tuple:
    """
    Worker function to process a single image in parallel.
    Loads image, computes blur score, and histogram if not blurry.

    Args:
        img_path_str: Path to image file as string

    Returns:
        Tuple of (img_path_str, blur_score, histogram or None, is_blurry)
    """
    frame = cv2.imread(img_path_str)
    if frame is None:
        return (img_path_str, 0.0, None, True)  # Treat unreadable as blurry

    blur_score = variance_of_laplacian(frame)

    if blur_score < _worker_blur_threshold:
        return (img_path_str, blur_score, None, True)

    # Compute histogram only for non-blurry images
    hist = compute_histogram(frame, _worker_bins)
    return (img_path_str, blur_score, hist, False)


def compute_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute normalized HSV histogram for a frame."""
    small_frame = frame[::8, ::8]
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    return np.concatenate([hist_h, hist_s, hist_v])


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compare two histograms using correlation method."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# ============================================================================
# CLUSTERING METHODS
# ============================================================================

def _import_dbscan():
    """Import sklearn DBSCAN, raising a clear error if unavailable."""
    try:
        from sklearn.cluster import DBSCAN
        return DBSCAN
    except ImportError:
        raise ImportError(
            "scikit-learn is required for clustering mode. "
            "Install it with: pip install scikit-learn"
        )


def cluster_sequential_frames(
    histograms: np.ndarray,
    eps: float = 0.15,
    window_size: int = 10
) -> np.ndarray:
    """
    Cluster sequential frames using a sliding window approach.

    For video frames, consecutive frames are most likely to be similar
    (stationary camera). This approach only compares each frame to its
    nearby neighbors, making it O(n × window_size) instead of O(n²).

    Frames are grouped into clusters by chaining: if frame A is similar
    to frame B, and frame B is similar to frame C, they're all in the
    same cluster even if A and C aren't directly similar.

    Args:
        histograms: Array of shape (N, D) with histogram feature vectors.
        eps: Max distance threshold. Frames with distance < eps are similar.
        window_size: Number of previous frames to compare against.

    Returns:
        Array of cluster labels, shape (N,). Cluster IDs start at 0.
        Unlike DBSCAN, there are no noise points (-1) — every frame
        belongs to a cluster (possibly a singleton cluster).
    """
    n = len(histograms)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([0], dtype=np.int32)

    # Union-Find for clustering
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare each frame to previous frames within the window
    histograms_f32 = histograms.astype(np.float32)
    for i in range(1, n):
        start = max(0, i - window_size)
        for j in range(start, i):
            corr = cv2.compareHist(histograms_f32[i], histograms_f32[j], cv2.HISTCMP_CORREL)
            dist = 1.0 - corr
            if dist < eps:
                union(i, j)

    # Convert union-find to cluster labels
    root_to_label = {}
    labels = np.zeros(n, dtype=np.int32)
    next_label = 0
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    return labels


def prompt_clustering_method() -> str:
    """
    Prompt user to choose clustering method.

    Returns:
        'sequential' or 'dbscan'
    """
    print("\nSelect clustering method:")
    print("  1. Sequential (fast) - Compares nearby frames only. Best for video frames.")
    print("  2. DBSCAN (thorough) - Compares all frames. Best for mixed/shuffled images.")
    while True:
        choice = input(f"{GREEN}Enter choice (1 or 2) [default: 1]: {RESET}").strip()
        if choice == "" or choice == "1":
            return "sequential"
        elif choice == "2":
            return "dbscan"
        else:
            print(f"{RED}Invalid choice. Please enter 1 or 2.{RESET}")


def build_histogram_distance_matrix(histograms: np.ndarray) -> np.ndarray:
    """
    Build a pairwise distance matrix from histogram feature vectors.

    Distance = 1.0 - correlation, so identical frames have distance 0.

    Args:
        histograms: Array of shape (N, D) with histogram feature vectors.

    Returns:
        Symmetric distance matrix of shape (N, N).
    """
    n = len(histograms)
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            corr = cv2.compareHist(
                histograms[i].astype(np.float32),
                histograms[j].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            dist = 1.0 - corr
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def cluster_frames_dbscan(
    histograms: np.ndarray,
    eps: float = 0.15,
    min_samples: int = 2
) -> np.ndarray:
    """
    Cluster frame histograms using DBSCAN and return cluster labels.

    Args:
        histograms: Array of shape (N, D).
        eps: Max distance between two samples in the same neighborhood.
        min_samples: Min samples in a neighborhood to form a core point.

    Returns:
        Array of cluster labels, shape (N,). -1 = noise/singleton.
    """
    DBSCAN = _import_dbscan()
    distance_matrix = build_histogram_distance_matrix(histograms)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    return model.fit_predict(distance_matrix)


def select_representatives_from_clusters(
    labels: np.ndarray,
    sharpness_scores: np.ndarray,
    frames: list,
    frame_indices: list
) -> list:
    """
    Select the sharpest frame from each cluster as the representative.

    Args:
        labels: Cluster labels from DBSCAN (shape N). -1 = noise/singleton.
        sharpness_scores: Laplacian variance for each frame (shape N).
        frames: List of frame pixel data (numpy arrays).
        frame_indices: Original frame indices for temporal ordering.

    Returns:
        List of (frame, original_index, sharpness) tuples for selected representatives,
        sorted by original frame index.
    """
    representatives = []

    for label in sorted(set(labels)):
        if label == -1:
            # Noise points: each is its own representative
            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices:
                representatives.append((frames[idx], frame_indices[idx], sharpness_scores[idx]))
            continue

        cluster_indices = np.where(labels == label)[0]
        best_idx = cluster_indices[np.argmax(sharpness_scores[cluster_indices])]
        representatives.append((frames[best_idx], frame_indices[best_idx], sharpness_scores[best_idx]))

    # Maintain temporal order
    representatives.sort(key=lambda x: x[1])
    return representatives


# ============================================================================
# USER INTERACTION UTILITIES
# ============================================================================

def prompt_with_default_value(prompt_text: str, default_value) -> str:
    """Prompt user for input, showing default value. Returns default if empty input."""
    default_str = str(default_value) if default_value is not None else ""
    if default_str:
        user_input = input(f"{GREEN}{prompt_text} [default: {default_str}]: {RESET}").strip()
    else:
        user_input = input(f"{GREEN}{prompt_text}: {RESET}").strip()
    return user_input if user_input else default_str


# ============================================================================
# VIDEO FILE DISCOVERY
# ============================================================================

def find_image_files(directory: str, extensions: list = None) -> list:
    """
    Find all image files in a directory and its subdirectories.

    Args:
        directory: Path to the directory to search
        extensions: List of image file extensions to look for

    Returns:
        Sorted list of Path objects for found image files
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']

    dir_path = Path(directory)
    image_files = []

    for ext in extensions:
        image_files.extend(dir_path.glob(f"**/*.{ext}"))
        image_files.extend(dir_path.glob(f"**/*.{ext.upper()}"))

    image_files.sort()
    return image_files


def find_video_files(directory: str, extensions: list = None) -> list:
    """
    Find all video files in a directory and its subdirectories.

    Args:
        directory: Path to the directory to search
        extensions: List of video file extensions to look for

    Returns:
        List of Path objects for found video files
    """
    if extensions is None:
        extensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v', 'mpeg', 'mpg', '3gp']

    dir_path = Path(directory)
    video_files = []

    for ext in extensions:
        video_files.extend(dir_path.glob(f"**/*.{ext}"))
        video_files.extend(dir_path.glob(f"**/*.{ext.upper()}"))

    # Sort by path for consistent ordering
    video_files.sort()
    return video_files


# ============================================================================
# FRAME EXTRACTION - SEQUENTIAL MODE
# ============================================================================

def extract_frames_from_single_video(
    video_path: str,
    output_dir: str,
    threshold: float = 0.85,
    target_fps: float = 3.0,
    max_frames: Optional[int] = None,
    bins: int = 32,
    prefix: str = "frame",
    blur_threshold: float = 100.0
) -> dict:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        threshold: Similarity threshold for frame deduplication
        target_fps: Target frames per second to extract
        max_frames: Maximum number of frames to extract (None for no limit)
        bins: Histogram bins for similarity comparison
        prefix: Filename prefix for extracted frames
        blur_threshold: Laplacian variance threshold for blur detection

    Returns:
        Dictionary with extraction statistics
    """
    import time
    start_time = time.time()

    video_info = get_video_info(video_path)
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    duration = video_info['duration']
    total_frames = video_info['total_frames']

    frame_interval = max(1, int(round(fps / target_fps)))
    actual_target_fps = fps / frame_interval
    expected_frames = int(duration * actual_target_fps) if duration > 0 else int(total_frames / frame_interval)

    print(f"{YELLOW}Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Frame interval: {frame_interval}")
    print(f"Similarity threshold: {threshold}")
    print(f"Blur threshold: {blur_threshold} (frames below this are skipped){RESET}")
    print("-" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start ffmpeg streaming
    vf_filter = f'fps={actual_target_fps}'
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', vf_filter,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-',
        '-hide_banner',
        '-loglevel', 'error'
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )

    frame_size = width * height * 3
    saved_frames = []
    last_histogram = None
    saved_count = 0
    processed_count = 0
    blurry_count = 0
    similar_count = 0

    # Track the best (sharpest) frame in a group of similar frames
    best_frame_in_group = None
    best_blur_score_in_group = -1
    group_size = 0

    def save_best_frame_from_group():
        """Save the sharpest frame from the current group of similar frames."""
        nonlocal saved_count, blurry_count, best_frame_in_group, best_blur_score_in_group, group_size

        if best_frame_in_group is None:
            return False

        # Check if the best frame passes blur threshold
        if best_blur_score_in_group < blur_threshold:
            # Entire group is too blurry
            blurry_count += 1
            best_frame_in_group = None
            best_blur_score_in_group = -1
            group_size = 0
            return False

        # Save the sharpest frame from the group
        output_filename = f"{prefix}_{saved_count:06d}.png"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), best_frame_in_group)
        saved_frames.append(str(output_file))
        saved_count += 1

        # Reset group tracking
        best_frame_in_group = None
        best_blur_score_in_group = -1
        group_size = 0
        return True

    try:
        with tqdm(total=expected_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                processed_count += 1

                current_hist = compute_histogram(frame, bins)
                blur_score = variance_of_laplacian(frame)

                # Check if this frame is unique (different from last SAVED frame)
                is_unique = last_histogram is None
                similarity = 0.0

                if not is_unique:
                    similarity = compare_histograms(last_histogram, current_hist)
                    is_unique = similarity < threshold

                if is_unique:
                    # New unique frame detected - save the best from previous group first
                    save_best_frame_from_group()

                    # Start a new group with this frame
                    best_frame_in_group = frame.copy()
                    best_blur_score_in_group = blur_score
                    group_size = 1
                    # Only update last_histogram when frame is unique (matches v4 behavior)
                    last_histogram = current_hist
                else:
                    # Similar frame - check if it's sharper than current best in group
                    similar_count += 1
                    group_size += 1
                    if blur_score > best_blur_score_in_group:
                        best_frame_in_group = frame.copy()
                        best_blur_score_in_group = blur_score

                pbar.set_postfix({'saved': saved_count, 'similar': similar_count, 'blurry': blurry_count, 'grp': group_size})

                if max_frames and saved_count >= max_frames:
                    print(f"\nReached maximum frame limit: {max_frames}")
                    break

                pbar.update(1)

            # Save the last group after loop ends
            save_best_frame_from_group()

    finally:
        process.stdout.close()
        process.terminate()
        process.wait()

    total_time = time.time() - start_time

    print("-" * 60)
    print(f"{CYAN}Completed!")
    print(f"Processed: {processed_count} frames")
    print(f"Similar (skipped): {similar_count} frames")
    print(f"Blurry (skipped): {blurry_count} frames")
    print(f"Saved: {saved_count} unique frames")
    print(f"Output directory: {output_dir}")
    print(f"Total time: {total_time:.2f}s{RESET}")

    return {
        'processed': processed_count,
        'similar': similar_count,
        'blurry': blurry_count,
        'saved': saved_count,
        'time': total_time,
        'output_dir': output_dir
    }


# ============================================================================
# FRAME EXTRACTION - CLUSTERING MODE
# ============================================================================

def extract_frames_clustering(
    video_path: str,
    output_dir: str,
    threshold: float = 0.85,
    target_fps: float = 3.0,
    max_frames: Optional[int] = None,
    bins: int = 32,
    prefix: str = "frame",
    blur_threshold: float = 100.0,
    eps: Optional[float] = None,
    min_samples: int = 2
) -> dict:
    """
    Extract frames from video using DBSCAN clustering-based deduplication.

    Two-pass approach:
      Pass 1: Stream frames, compute histograms and sharpness, filter blurry.
      Pass 2: Cluster histograms with DBSCAN, select sharpest per cluster, save.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        threshold: Similarity threshold (used to auto-derive eps if eps is None)
        target_fps: Target frames per second to extract
        max_frames: Maximum number of frames to extract (None for no limit)
        bins: Histogram bins for similarity comparison
        prefix: Filename prefix for extracted frames
        blur_threshold: Laplacian variance threshold for blur detection
        eps: DBSCAN eps parameter; None = auto-derived as 1.0 - threshold
        min_samples: DBSCAN min_samples parameter

    Returns:
        Dictionary with extraction statistics
    """
    import time
    start_time = time.time()

    if eps is None:
        eps = 1.0 - threshold

    video_info = get_video_info(video_path)
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    duration = video_info['duration']
    total_frames = video_info['total_frames']

    frame_interval = max(1, int(round(fps / target_fps)))
    actual_target_fps = fps / frame_interval
    expected_frames = int(duration * actual_target_fps) if duration > 0 else int(total_frames / frame_interval)

    print(f"{YELLOW}Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Mode: DBSCAN Clustering (eps={eps:.4f}, min_samples={min_samples})")
    print(f"Blur threshold: {blur_threshold}{RESET}")
    print("-" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: Collect candidate frames ----
    vf_filter = f'fps={actual_target_fps}'
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', vf_filter,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-',
        '-hide_banner',
        '-loglevel', 'error'
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )

    frame_size = width * height * 3
    candidate_frames = []
    candidate_histograms = []
    candidate_sharpness = []
    candidate_indices = []
    processed_count = 0
    blurry_count = 0

    try:
        with tqdm(total=expected_frames, desc="Pass 1: Collecting frames", unit="frame") as pbar:
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                processed_count += 1

                blur_score = variance_of_laplacian(frame)

                if blur_score < blur_threshold:
                    blurry_count += 1
                    pbar.update(1)
                    continue

                hist = compute_histogram(frame, bins)

                candidate_frames.append(frame.copy())
                candidate_histograms.append(hist)
                candidate_sharpness.append(blur_score)
                candidate_indices.append(processed_count)

                pbar.set_postfix({'candidates': len(candidate_frames), 'blurry': blurry_count})
                pbar.update(1)
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()

    print(f"\nPass 1 complete: {len(candidate_frames)} candidates from {processed_count} frames "
          f"({blurry_count} blurry skipped)")

    # Edge cases
    if len(candidate_frames) == 0:
        print(f"{RED}No candidate frames after blur filtering.{RESET}")
        return {
            'processed': processed_count, 'similar': 0, 'blurry': blurry_count,
            'saved': 0, 'time': time.time() - start_time, 'output_dir': output_dir
        }

    if len(candidate_frames) == 1:
        output_file = output_path / f"{prefix}_000000.png"
        cv2.imwrite(str(output_file), candidate_frames[0])
        return {
            'processed': processed_count, 'similar': 0, 'blurry': blurry_count,
            'saved': 1, 'time': time.time() - start_time, 'output_dir': output_dir
        }

    # ---- Pass 2: Cluster and select representatives ----
    print("Pass 2: Clustering frames with DBSCAN...")
    histograms_array = np.array(candidate_histograms)
    sharpness_array = np.array(candidate_sharpness)

    labels = cluster_frames_dbscan(
        histograms=histograms_array,
        eps=eps,
        min_samples=min_samples
    )

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    print(f"Clustering result: {n_clusters} clusters, {n_noise} noise/singleton points")

    representatives = select_representatives_from_clusters(
        labels=labels,
        sharpness_scores=sharpness_array,
        frames=candidate_frames,
        frame_indices=candidate_indices
    )

    if max_frames and len(representatives) > max_frames:
        representatives = representatives[:max_frames]

    # Save representative frames
    saved_count = 0
    for frame_data, orig_idx, sharpness in representatives:
        output_filename = f"{prefix}_{saved_count:06d}.png"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), frame_data)
        saved_count += 1

    total_time = time.time() - start_time
    similar_count = len(candidate_frames) - saved_count

    print("-" * 60)
    print(f"{CYAN}Completed!")
    print(f"Processed: {processed_count} frames")
    print(f"Blurry (skipped): {blurry_count} frames")
    print(f"Clustered candidates: {len(candidate_frames)}")
    print(f"Saved: {saved_count} representative frames")
    print(f"Output directory: {output_dir}")
    print(f"Total time: {total_time:.2f}s{RESET}")

    return {
        'processed': processed_count,
        'similar': similar_count,
        'blurry': blurry_count,
        'saved': saved_count,
        'time': total_time,
        'output_dir': output_dir
    }


# ============================================================================
# IMAGE DIRECTORY CLUSTERING
# ============================================================================

def cluster_image_directory(
    input_dir: str,
    output_dir: str,
    deleted_dir: str = "./deleted/clustering",
    blur_threshold: float = 100.0,
    threshold: float = 0.85,
    eps: Optional[float] = None,
    min_samples: int = 2,
    bins: int = 32,
    image_extensions: list = None,
    clustering_method: Optional[str] = None,
    window_size: int = 10
) -> dict:
    """
    Apply clustering and blur filtering to an image directory.

    Loads all images, filters blurry ones, clusters the rest by visual
    similarity, and keeps only the sharpest representative per cluster.
    Non-representative and blurry images are moved to the deleted directory.

    Args:
        input_dir: Directory containing images to cluster
        output_dir: Directory to copy representative images to
        deleted_dir: Directory to move non-representative/blurry images to
        blur_threshold: Laplacian variance threshold for blur detection
        threshold: Similarity threshold (used to auto-derive eps if eps is None)
        eps: DBSCAN eps parameter; None = auto-derived as 1.0 - threshold
        min_samples: DBSCAN min_samples parameter
        bins: Histogram bins for similarity comparison
        image_extensions: List of image file extensions to process
        clustering_method: 'sequential' or 'dbscan'. If None, prompts user.
        window_size: For sequential method, number of neighboring frames to compare.

    Returns:
        Dictionary with clustering statistics
    """
    import time
    start_time = time.time()

    if eps is None:
        eps = 1.0 - threshold

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    deleted_path = Path(deleted_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    deleted_path.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(input_dir, image_extensions)

    if not image_files:
        print(f"{RED}No image files found in: {input_dir}{RESET}")
        return {
            'processed': 0, 'blurry': 0, 'clustered': 0,
            'saved': 0, 'moved': 0, 'time': 0, 'output_dir': output_dir
        }

    print(f"{YELLOW}Input directory: {input_dir}")
    print(f"Found {len(image_files)} images{RESET}")

    # Prompt user for clustering method if not specified
    if clustering_method is None:
        clustering_method = prompt_clustering_method()

    if clustering_method == "sequential":
        print(f"{YELLOW}Mode: Sequential Clustering (window={window_size}, eps={eps:.4f})")
    else:
        print(f"{YELLOW}Mode: DBSCAN Clustering (eps={eps:.4f}, min_samples={min_samples})")
    print(f"Blur threshold: {blur_threshold}{RESET}")
    print("-" * 60)

    # Group images by parent directory (cluster per-folder for efficiency)
    from collections import defaultdict
    folder_groups = defaultdict(list)
    for img_path in image_files:
        folder_groups[img_path.parent].append(img_path)

    n_folders = len(folder_groups)
    print(f"{CYAN}Images spread across {n_folders} folder(s){RESET}")
    print("-" * 60)

    # Process each folder independently
    total_processed = 0
    total_blurry = 0
    total_candidates = 0
    saved_count = 0
    moved_count = 0

    # Determine number of worker processes
    n_workers = min(cpu_count(), 8)  # Cap at 8 to avoid too much overhead
    print(f"Using {n_workers} parallel workers for image processing")

    for folder_idx, (folder, folder_files) in enumerate(sorted(folder_groups.items()), 1):
        folder_label = folder.relative_to(input_path) if folder.is_relative_to(input_path) else folder
        if n_folders > 1:
            print(f"\n[Folder {folder_idx}/{n_folders}] {folder_label}/ ({len(folder_files)} images)")

        # Pass 1: Load images, compute features, filter blurry - IN PARALLEL
        candidate_files = []
        candidate_histograms = []
        candidate_sharpness = []

        # Convert paths to strings for multiprocessing
        img_paths_str = [str(p) for p in folder_files]

        with Pool(processes=n_workers, initializer=_init_worker, initargs=(blur_threshold, bins)) as pool:
            # Process images in parallel with progress bar
            results = list(tqdm(
                pool.imap(_process_single_image, img_paths_str),
                total=len(img_paths_str),
                desc="Analyzing images",
                unit="img"
            ))

        # Collect results
        for img_path_str, blur_score, hist, is_blurry in results:
            if is_blurry:
                if blur_score > 0:  # Was readable but blurry
                    total_blurry += 1
                    total_processed += 1
                # else: unreadable, skip silently
                continue

            total_processed += 1
            candidate_files.append(Path(img_path_str))
            candidate_histograms.append(hist)
            candidate_sharpness.append(blur_score)

        total_candidates += len(candidate_files)

        # Edge cases
        if len(candidate_files) == 0:
            continue

        if len(candidate_files) == 1:
            rel = candidate_files[0].relative_to(input_path) if candidate_files[0].is_relative_to(input_path) else Path(candidate_files[0].name)
            dest = output_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(candidate_files[0]), str(dest))
            saved_count += 1
            continue

        # Pass 2: Cluster and select representatives within this folder
        histograms_array = np.array(candidate_histograms)
        sharpness_array = np.array(candidate_sharpness)

        if clustering_method == "sequential":
            labels = cluster_sequential_frames(
                histograms=histograms_array,
                eps=eps,
                window_size=window_size
            )
            n_clusters = len(set(labels))
            print(f"  Clustering: {n_clusters} clusters (sequential)")
        else:
            labels = cluster_frames_dbscan(
                histograms=histograms_array,
                eps=eps,
                min_samples=min_samples
            )
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            print(f"  Clustering: {n_clusters} clusters, {n_noise} noise/singleton points")

        # Determine representative indices
        representative_indices = set()
        for label in sorted(set(labels)):
            if label == -1:
                # DBSCAN noise points - keep all of them
                noise_idx = np.where(labels == -1)[0]
                representative_indices.update(noise_idx.tolist())
                continue

            cluster_idx = np.where(labels == label)[0]
            best_idx = cluster_idx[np.argmax(sharpness_array[cluster_idx])]
            representative_indices.add(int(best_idx))

        # Copy only representatives to output - skip non-representatives entirely
        for i, img_path in enumerate(candidate_files):
            if i in representative_indices:
                rel = img_path.relative_to(input_path) if img_path.is_relative_to(input_path) else Path(img_path.name)
                dest = output_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(dest))
                saved_count += 1
            else:
                moved_count += 1  # Count as "skipped" but don't actually copy

    total_time = time.time() - start_time

    print(f"\n{CYAN}Completed!")
    print(f"Folders processed: {n_folders}")
    print(f"Processed: {total_processed} images")
    print(f"Blurry (moved to deleted): {total_blurry}")
    print(f"Clustered candidates: {total_candidates}")
    print(f"Saved: {saved_count} representative images")
    print(f"Moved to deleted: {moved_count}")
    print(f"Output directory: {output_dir}")
    print(f"Deleted directory: {deleted_dir}")
    print(f"Total time: {total_time:.2f}s{RESET}")

    return {
        'processed': total_processed,
        'blurry': total_blurry,
        'clustered': total_candidates,
        'saved': saved_count,
        'moved': moved_count,
        'time': total_time,
        'output_dir': output_dir
    }


# ============================================================================
# MAIN STEP ENTRY POINT
# ============================================================================

def extract_video_frames(config: Dict[str, Any], from_previous_step: bool = False) -> bool:
    """Step 1: Extract and deduplicate frames from video(s) or image directories.

    Supports:
    - Single video file → extract frames with DBSCAN clustering + blur filtering
    - Directory with videos → extract frames from each video with clustering
    - Directory with images (no videos) → apply DBSCAN clustering + blur filtering

    Clustering and blur detection are always applied regardless of input type.
    """
    input_path_str = config.get("video_path", "")
    threshold = config.get("frame_threshold", 0.85)
    target_fps = config.get("target_fps", 3.0)
    max_frames = config.get("max_frames")
    bins = config.get("histogram_bins", 32)
    prefix = config.get("frame_prefix", "frame")
    blur_threshold = config.get("blur_threshold", 100.0)
    clustering_eps = config.get("clustering_eps")
    clustering_min_samples = config.get("clustering_min_samples", 2)
    clustering_deleted_dir = config.get("clustering_deleted_dir", "./deleted/clustering")
    clustering_method = config.get("clustering_method")  # None = prompt user
    clustering_window_size = config.get("clustering_window_size", 10)
    image_extensions = config.get("image_extensions", ["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

    if not input_path_str:
        input_path_str = prompt_with_default_value(
            "Enter video file, image folder, or video folder path",
            config.get("video_path", "")
        )
        if not input_path_str:
            print(f"{RED}No path provided.{RESET}")
            return False

    if not os.path.exists(input_path_str):
        print(f"{RED}Error: Path '{input_path_str}' does not exist.{RESET}")
        return False

    path = Path(input_path_str)

    if path.is_dir():
        # Directory mode: check for videos first, then images
        video_files = find_video_files(input_path_str)

        if video_files:
            # Directory contains videos → extract frames with clustering
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                print(f"{RED}Error: ffmpeg/ffprobe not found. Please install ffmpeg.{RESET}")
                return False

            print(f"Scanning directory for videos: {input_path_str}")
            print(f"Found {len(video_files)} video(s):\n")
            for i, vf in enumerate(video_files, 1):
                print(f"  {i}. {vf.relative_to(path) if vf.is_relative_to(path) else vf.name}")
            print()

            base_name = path.name
            base_output_dir = f"./extracted_frames/{base_name}"

            total_stats = {
                'videos_processed': 0,
                'videos_failed': 0,
                'total_saved': 0,
                'total_processed': 0,
                'total_time': 0
            }

            for idx, video_file in enumerate(video_files, 1):
                print(f"\n{'=' * 60}")
                print(f"{CYAN}Processing video {idx}/{len(video_files)}: {video_file.name}{RESET}")
                print("=" * 60)

                video_name = video_file.stem
                output_dir = f"{base_output_dir}/{video_name}"

                try:
                    stats = extract_frames_clustering(
                        video_path=str(video_file),
                        output_dir=output_dir,
                        threshold=threshold,
                        target_fps=target_fps,
                        max_frames=max_frames,
                        bins=bins,
                        prefix=prefix,
                        blur_threshold=blur_threshold,
                        eps=clustering_eps,
                        min_samples=clustering_min_samples
                    )
                    total_stats['videos_processed'] += 1
                    total_stats['total_saved'] += stats['saved']
                    total_stats['total_processed'] += stats['processed']
                    total_stats['total_time'] += stats['time']
                except Exception as e:
                    print(f"{RED}Error processing {video_file.name}: {e}{RESET}")
                    total_stats['videos_failed'] += 1

            print("\n" + "=" * 60)
            print("BATCH EXTRACTION SUMMARY")
            print("=" * 60)
            print(f"{CYAN}Videos processed: {total_stats['videos_processed']}")
            print(f"Videos failed: {total_stats['videos_failed']}")
            print(f"Total frames processed: {total_stats['total_processed']}")
            print(f"Total frames saved: {total_stats['total_saved']}")
            print(f"Total time: {total_stats['total_time']:.2f}s")
            print(f"Output directory: {base_output_dir}{RESET}")

            config["extracted_frames_dir"] = base_output_dir
            config["video_name"] = base_name

        else:
            # No videos found — check for images
            image_files = find_image_files(input_path_str, image_extensions)

            if not image_files:
                print(f"{RED}No video or image files found in: {input_path_str}{RESET}")
                return False

            print(f"No videos found. Found {len(image_files)} image(s) — applying clustering.")

            base_name = path.name
            output_dir = f"./extracted_frames/{base_name}"

            try:
                stats = cluster_image_directory(
                    input_dir=input_path_str,
                    output_dir=output_dir,
                    deleted_dir=clustering_deleted_dir,
                    blur_threshold=blur_threshold,
                    threshold=threshold,
                    eps=clustering_eps,
                    min_samples=clustering_min_samples,
                    bins=bins,
                    image_extensions=image_extensions,
                    clustering_method=clustering_method,
                    window_size=clustering_window_size
                )
            except Exception as e:
                print(f"{RED}Error clustering images: {e}{RESET}")
                return False

            config["extracted_frames_dir"] = output_dir
            config["video_name"] = base_name

    else:
        # Single file mode (must be a video)
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            print(f"{RED}Error: ffmpeg/ffprobe not found. Please install ffmpeg.{RESET}")
            return False

        video_name = path.stem
        output_dir = f"./extracted_frames/{video_name}"

        try:
            extract_frames_clustering(
                video_path=str(path),
                output_dir=output_dir,
                threshold=threshold,
                target_fps=target_fps,
                max_frames=max_frames,
                bins=bins,
                prefix=prefix,
                blur_threshold=blur_threshold,
                eps=clustering_eps,
                min_samples=clustering_min_samples
            )
        except Exception as e:
            print(f"{RED}Error processing video: {e}{RESET}")
            return False

        config["extracted_frames_dir"] = output_dir
        config["video_name"] = video_name

    return True
