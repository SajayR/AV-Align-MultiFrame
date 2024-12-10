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
    def __init__(self, sample_rate=16000, n_mels=128, target_length=300):
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

def clean_videos(video_dir: str, move_bad=True):
    """
    Check and clean videos with audio issues. 
    If move_bad=True, moves bad videos to a 'bad_videos' directory.
    If move_bad=False, deletes bad videos.
    """
    print(f"Checking audio in directory: {video_dir}")
    video_dir = Path(video_dir)
    paths = sorted(video_dir.glob("*.mp4"))
    checker = AudioChecker()
    
    # Create directory for bad videos if moving them
    if move_bad:
        bad_dir = video_dir / "bad_videos"
        bad_dir.mkdir(exist_ok=True)
    
    bad_videos = []
    error_types = {}
    vid_segments = {}  # Track good/bad segments per vid_num
    
    for path in tqdm(paths, desc="Checking videos"):
        vid_num = int(path.stem.split('_')[0])
        segment = int(path.stem.split('_')[1]) if '_' in path.stem else 0
        
        if vid_num not in vid_segments:
            vid_segments[vid_num] = {'total': 0, 'bad': 0}
        vid_segments[vid_num]['total'] += 1
        
        try:
            # Try to load and process audio
            container = av.open(str(path))
            
            if not container.streams.audio:
                bad_videos.append((path, "No audio stream"))
                error_types["No audio stream"] = error_types.get("No audio stream", 0) + 1
                vid_segments[vid_num]['bad'] += 1
                continue
                
            audio = container.streams.audio[0]
            audio_frames = []
            
            for frame in container.decode(audio=0):
                audio_frames.append(frame.to_ndarray())
            
            if not audio_frames:
                bad_videos.append((path, "No audio frames"))
                error_types["No audio frames"] = error_types.get("No audio frames", 0) + 1
                vid_segments[vid_num]['bad'] += 1
                continue
                
            waveform = torch.from_numpy(np.concatenate(audio_frames))
            
            if waveform.abs().max() < 1e-6:
                bad_videos.append((path, "Silent audio"))
                error_types["Silent audio"] = error_types.get("Silent audio", 0) + 1
                vid_segments[vid_num]['bad'] += 1
                continue
            
            mel_spec, error_msg = checker.process_audio(waveform)
            if mel_spec is None:
                bad_videos.append((path, error_msg))
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
                vid_segments[vid_num]['bad'] += 1
            
        except Exception as e:
            error_msg = str(e)[:100]
            bad_videos.append((path, f"Error: {error_msg}"))
            error_types[f"Error: {error_msg}"] = error_types.get(f"Error: {error_msg}", 0) + 1
            vid_segments[vid_num]['bad'] += 1
        
        container.close()
    
    # Handle bad videos
    for path, error in bad_videos:
        if move_bad:
            shutil.move(str(path), str(bad_dir / path.name))
        else:
            path.unlink()
    
    # Print summary
    print("\n=== Audio Check Summary ===")
    print(f"Total videos checked: {len(paths)}")
    print(f"Bad segments removed: {len(bad_videos)}")
    
    print("\nError types distribution:")
    for error_type, count in error_types.items():
        print(f"{error_type}: {count}")
    
    print("\nVid nums with partially bad segments:")
    partial_bad = {vid: stats for vid, stats in vid_segments.items() 
                  if 0 < stats['bad'] < stats['total']}
    for vid, stats in partial_bad.items():
        print(f"Vid {vid}: {stats['bad']}/{stats['total']} segments bad")
    
    # Save results
    output_dir = video_dir / "cleaning_results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "cleaning_summary.txt", "w") as f:
        f.write(f"Total videos checked: {len(paths)}\n")
        f.write(f"Bad segments removed: {len(bad_videos)}\n\n")
        f.write("Bad segments details:\n")
        for path, error in bad_videos:
            f.write(f"{path.name}: {error}\n")
    
    print(f"\nResults saved to {output_dir}/")
    return vid_segments

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", help="Directory containing the video files")
    parser.add_argument("--move", action="store_true", 
                       help="Move bad videos to 'bad_videos' directory instead of deleting")
    args = parser.parse_args()
    
    vid_segments = clean_videos(args.video_dir, move_bad=args.move)