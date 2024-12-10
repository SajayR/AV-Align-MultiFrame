import torch
import torchaudio
import av
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torchaudio.transforms as T
import shutil

class AudioChecker:
    def __init__(self, sample_rate=16000, n_mels=128, target_length=998):  # Updated for 10s
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.025 * sample_rate),
            hop_length=int(0.010 * sample_rate),
            n_mels=n_mels,
            power=2.0,
            normalized=True
        )
    
    def process_audio(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        mel = self.mel_spec(waveform)
        mel = torch.log(mel + 1e-10)
        
        mel_mean = mel.mean()
        mel_std = mel.std()
        
        if mel_std == 0:
            return None, "Zero standard deviation in mel spectrogram"
            
        mel = (mel - mel_mean) / (mel_std * 2)
        
        current_length = mel.shape[2]
        if current_length > self.target_length:
            start = (current_length - self.target_length) // 2
            mel = mel[:, :, start:start + self.target_length]
        else:
            repeats = (self.target_length + current_length - 1) // current_length
            mel = mel.repeat(1, 1, repeats)
            mel = mel[:, :, :self.target_length]
        
        mel = mel.squeeze(0).t()
        
        if torch.isnan(mel).any():
            return None, "NaN values in mel spectrogram"
        
        return mel, None

def check_duration(container):
    """Check if video duration is exactly 10 seconds"""
    # Get duration in seconds
    duration = float(container.duration) / float(av.time_base)
    
    # Allow for a small tolerance (±0.1 seconds)
    tolerance = 0.1
    if abs(duration - 10.0) > tolerance:
        return False, f"Duration is {duration:.2f}s (expected 10s)"
    return True, None

def clean_videos(video_dir: str, move_bad=True):
    """
    Check and clean videos with audio issues and incorrect duration.
    If move_bad=True, moves bad videos to a 'bad_videos' directory.
    If move_bad=False, deletes bad videos.
    """
    print(f"Checking videos in directory: {video_dir}")
    video_dir = Path(video_dir)
    paths = sorted(video_dir.glob("*.mp4"))
    
    # Print total number of videos
    total_videos = len(paths)
    print(f"\nTotal number of videos found: {total_videos}")
    
    checker = AudioChecker()
    
    # Create directory for bad videos if moving them
    if move_bad:
        bad_dir = video_dir / "bad_videos"
        bad_dir.mkdir(exist_ok=True)
    
    bad_videos = []
    error_types = {}
    duration_stats = {"too_short": 0, "too_long": 0, "exact": 0}
    
    for path in tqdm(paths, desc="Checking videos"):
        try:
            # Try to load and process video
            container = av.open(str(path))
            
            # Check duration first
            duration_ok, duration_error = check_duration(container)
            if not duration_ok:
                bad_videos.append((path, duration_error))
                error_types[duration_error] = error_types.get(duration_error, 0) + 1
                # Update duration stats
                duration = float(container.duration) / float(av.time_base)
                if duration < 10:
                    duration_stats["too_short"] += 1
                else:
                    duration_stats["too_long"] += 1
                container.close()
                continue
            
            duration_stats["exact"] += 1
            
            if not container.streams.audio:
                bad_videos.append((path, "No audio stream"))
                error_types["No audio stream"] = error_types.get("No audio stream", 0) + 1
                continue
                
            audio = container.streams.audio[0]
            audio_frames = []
            
            for frame in container.decode(audio=0):
                audio_frames.append(frame.to_ndarray())
            
            if not audio_frames:
                bad_videos.append((path, "No audio frames"))
                error_types["No audio frames"] = error_types.get("No audio frames", 0) + 1
                continue
                
            waveform = torch.from_numpy(np.concatenate(audio_frames))
            
            if waveform.abs().max() < 1e-6:
                bad_videos.append((path, "Silent audio"))
                error_types["Silent audio"] = error_types.get("Silent audio", 0) + 1
                continue
            
            mel_spec, error_msg = checker.process_audio(waveform)
            if mel_spec is None:
                bad_videos.append((path, error_msg))
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
            
        except Exception as e:
            error_msg = str(e)[:100]
            bad_videos.append((path, f"Error: {error_msg}"))
            error_types[f"Error: {error_msg}"] = error_types.get(f"Error: {error_msg}", 0) + 1
        
        container.close()
    
    # Handle bad videos (commented out as requested)
    for path, error in tqdm(bad_videos, desc="Moving bad videos"):
        if move_bad:
            shutil.move(str(path), str(bad_dir / path.name))
        else:
            path.unlink()
    
    # Print summary
    print("\n=== Video Check Summary ===")
    print(f"Total videos: {total_videos}")
    print(f"Good videos: {total_videos - len(bad_videos)}")
    print(f"Bad videos: {len(bad_videos)}")
    
    print("\nDuration Statistics:")
    print(f"Exactly 10s (±0.1s): {duration_stats['exact']}")
    print(f"Too short (<10s): {duration_stats['too_short']}")
    print(f"Too long (>10s): {duration_stats['too_long']}")
    
    print("\nError types distribution:")
    for error_type, count in error_types.items():
        print(f"{error_type}: {count}")
    
    # Save results
    output_dir = video_dir / "cleaning_results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "cleaning_summary.txt", "w") as f:
        f.write(f"Total videos: {total_videos}\n")
        f.write(f"Good videos: {total_videos - len(bad_videos)}\n")
        f.write(f"Bad videos: {len(bad_videos)}\n\n")
        f.write("Duration Statistics:\n")
        f.write(f"Exactly 10s (±0.1s): {duration_stats['exact']}\n")
        f.write(f"Too short (<10s): {duration_stats['too_short']}\n")
        f.write(f"Too long (>10s): {duration_stats['too_long']}\n\n")
        f.write("Error types distribution:\n")
        for error_type, count in error_types.items():
            f.write(f"{error_type}: {count}\n")
        f.write("\nBad videos details:\n")
        for path, error in bad_videos:
            f.write(f"{path.name}: {error}\n")
    
    print(f"\nResults saved to {output_dir}/")
    return bad_videos

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", help="Directory containing the video files", default="/home/cisco/heyo/densefuck/VGGSound/vggsound_download/downloads")
    parser.add_argument("--move", action="store_true", 
                       help="Move bad videos to 'bad_videos' directory instead of deleting")
    args = parser.parse_args()
    
    bad_videos = clean_videos(args.video_dir, move_bad=args.move)