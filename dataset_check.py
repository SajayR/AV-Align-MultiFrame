from pathlib import Path
import torchvision
import numpy as np
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_video_fps(video_path):
    try:
        _, _, meta = torchvision.io.read_video(str(video_path), pts_unit='sec')
        return (video_path, meta["video_fps"])
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return (video_path, None)

def analyze_fps_distribution(delete_low_fps=False, fps_threshold=20):
    video_dir = Path("/home/cisco/heyo/densefuck/sound_of_pixels/densetok/densefuckfuckfuck/vggsound_split_1seconds")
    
    print("Starting analysis")
    video_files = list(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} MP4 files")
    
    # Use multiprocessing
    num_processes = cpu_count()  # Use all available CPU cores
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(get_video_fps, video_files),
            total=len(video_files),
            desc="Processing videos"
        ))
    
    # Separate paths and fps values, handle deletions
    low_fps_videos = []
    fps_values = []
    
    for path, fps in results:
        if fps is not None:
            if fps < fps_threshold:
                low_fps_videos.append(path)
            else:
                fps_values.append(fps)
    
    if delete_low_fps:
        print(f"\nFound {len(low_fps_videos)} videos below {fps_threshold} FPS")
        if input(f"Do you want to delete these {len(low_fps_videos)} videos? (y/n): ").lower() == 'y':
            for video_path in tqdm(low_fps_videos, desc="Deleting low FPS videos"):
                video_path.unlink()
            print(f"Deleted {len(low_fps_videos)} videos")
        else:
            print("Deletion cancelled")
    
    # Analysis of remaining videos
    fps_array = np.array(fps_values)
    
    print("\nFPS Distribution Statistics:")
    print(f"Mean FPS: {np.mean(fps_array):.2f}")
    print(f"Median FPS: {np.median(fps_array):.2f}")
    print(f"Min FPS: {np.min(fps_array):.2f}")
    print(f"Max FPS: {np.max(fps_array):.2f}")
    print(f"Standard Deviation: {np.std(fps_array):.2f}")
    
    fps_counter = Counter(fps_values)
    print("\nFPS Value Frequencies:")
    for fps, count in sorted(fps_counter.items()):
        percentage = (count / len(fps_values)) * 100
        print(f"FPS: {fps:.2f} - Count: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_fps_distribution(delete_low_fps=True, fps_threshold=20)