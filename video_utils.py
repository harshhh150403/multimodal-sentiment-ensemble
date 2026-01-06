"""
Video processing utilities for the Multimodal Sentiment Analysis project.

Contains functions for extracting and visualizing video frames.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from config import MAX_FRAMES, FRAME_SIZE


def extract_video_frames(video_path, max_frames=MAX_FRAMES):
    """
    Extract frames from video files at regular intervals.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        Numpy array of normalized frames with shape (max_frames, 224, 224, 3)
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        # Handle empty or invalid videos
        if total_frames == 0:
            return np.zeros((max_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        # Calculate which frame indices to extract
        if total_frames >= max_frames:
            # Sample frames evenly across the whole video
            frame_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
        else:
            # Take all available frames
            frame_indices = np.arange(total_frames)

        # Extract frames at calculated intervals
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize to target size
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                # If frame read fails, add zero frame
                frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32))

        # Pad with zero frames if we have fewer than max_frames
        if len(frames) < max_frames:
            padding_count = max_frames - len(frames)
            padding = [np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)] * padding_count
            frames.extend(padding)

        return np.array(frames)
    
    finally:
        # Always release the video capture
        if cap is not None:
            cap.release()


def visualize_video_frames(video_path, num_frames=5):
    """
    Visualize frames from a video file using matplotlib.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to display
    """
    # Extract frames
    frames = extract_video_frames(video_path, MAX_FRAMES)

    # Find actual (non-padded) frames
    actual_frames = []
    for frame in frames:
        # Check if frame is padding (all zeros)
        if np.all(frame == 0):
            break
        actual_frames.append(frame)

    # Select frames to display
    display_frames = actual_frames[:num_frames]
    
    if len(display_frames) == 0:
        print(f"No valid frames found in {video_path}")
        return

    # Create plot
    fig, axes = plt.subplots(1, len(display_frames), figsize=(4 * len(display_frames), 4))
    
    # Handle single frame case
    if len(display_frames) == 1:
        axes = [axes]

    # Plot each frame
    for idx, frame in enumerate(display_frames):
        axes[idx].imshow(frame)
        axes[idx].axis('off')
        axes[idx].set_title(f'Frame {idx + 1}')

    video_name = os.path.basename(video_path)
    plt.suptitle(f'First {len(display_frames)} Frames from {video_name}')
    plt.tight_layout()
    plt.show()
