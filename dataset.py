import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchaudio
import torchvision
import os
from pathlib import Path
import numpy as np
import random
import av
import matplotlib.pyplot as plt
from typing import Dict, List

class VideoBatchSampler(Sampler):
    def __init__(self, vid_nums: List[int], batch_size: int):
        self.vid_nums = np.array(vid_nums)
        self.batch_size = batch_size
        
        # Group indices by vid_num
        self.vid_to_indices = {}
        for i, vid in enumerate(vid_nums):
            if vid not in self.vid_to_indices:
                self.vid_to_indices[vid] = []
            self.vid_to_indices[vid].append(i)
            
    def __iter__(self):
        # Get unique vid_nums
        unique_vids = list(self.vid_to_indices.keys())
        random.shuffle(unique_vids)  # Shuffle at epoch start
        
        while len(unique_vids) >= self.batch_size:
            batch_vids = unique_vids[:self.batch_size]
            
            # For each selected video, randomly pick one of its segments
            batch = []
            for vid in batch_vids:
                idx = random.choice(self.vid_to_indices[vid])
                batch.append(idx)
            
            yield batch
            unique_vids = unique_vids[self.batch_size:]
    
    def __len__(self):
        return len(set(self.vid_nums)) // self.batch_size

class AudioVisualDataset(Dataset):
    def __init__(self, 
                 video_dir: str,
                 frame_transform=None,
                 audio_transform=None,
                 sample_rate: int = 16000,
                 num_frames: int = 1):
        
        self.video_paths = sorted([str(p) for p in Path(video_dir).glob("*.mp4")])
        self.frame_transform = frame_transform
        self.audio_transform = audio_transform
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        
        # Parse video numbers for creating negative pairs
        self.vid_nums = [int(os.path.basename(p).split('_')[0]) 
                        for p in self.video_paths]
    
    def _load_video_frame(self, video_path: str) -> torch.Tensor:
        try:
            container = av.open(video_path)
            container.streams.video[0].thread_type = "AUTO"
            
            # Get middle frame
            target_frame = container.streams.video[0].frames // 2
            
            for i, frame in enumerate(container.decode(video=0)):
                if i == target_frame:
                    # Convert to numpy array (H, W, 3)
                    numpy_frame = frame.to_ndarray(format='rgb24')
                    
                    # Convert to tensor, permute to (C, H, W) and convert to float32
                    frame_tensor = torch.from_numpy(numpy_frame).permute(2, 0, 1).float() / 255.0
                    
                    if self.frame_transform:
                        frame_tensor = self.frame_transform(frame_tensor)
                    return frame_tensor
                    
            container.close()
            
        except Exception as e:
            print(f"Error loading video frame from {video_path}: {str(e)}")
            raise
            
    def _load_audio(self, video_path: str) -> torch.Tensor:
        try:
            container = av.open(video_path)
            audio = container.streams.audio[0]
            
            audio_frames = []
            for frame in container.decode(audio=0):
                audio_frames.append(frame.to_ndarray())
            
            waveform = torch.from_numpy(np.concatenate(audio_frames))
            
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
            
            return waveform
        except Exception as e:
            print(f"Error loading audio from {video_path}: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_paths[idx]
        
        frame = self._load_video_frame(video_path)
        audio = self._load_audio(video_path)
        
        return {
            'frame': frame,
            'audio': audio,
            'vid_num': self.vid_nums[idx]
        }

if __name__ == "__main__":
    # Define transforms
    frame_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # We don't need ToPILImage() anymore since we're already handling the format conversion
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
    ])
    
    # Test the dataset
    dataset = AudioVisualDataset(
        video_dir='/home/cisco/heyo/densefuck/sound_of_pixels/dataset/solo_split_videos',
        frame_transform=frame_transform
    )
    
    # Test 1: Basic sample loading
    print("=== Test 1: Basic Sample Loading ===")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Frame shape: {sample['frame'].shape}")
    print(f"Audio shape: {sample['audio'].shape}")
    
    # Test 2: Visualize frame
    print("\n=== Test 2: Frame Visualization ===")
    plt.figure(figsize=(10, 10))
    frame = sample['frame'].permute(1, 2, 0).numpy()
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    plt.imshow(frame)
    plt.title(f"Frame from video {sample['vid_num']}")
    plt.show()
    
    # Test 3: Batch sampling
    print("\n=== Test 3: Batch Sampling ===")
    batch_size = 1024
    batch_sampler = VideoBatchSampler(dataset.vid_nums, batch_size=batch_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=2
    )
    
    # Get first batch and verify all vid_nums are different
    batch = next(iter(dataloader))
    vid_nums = batch['vid_num']
    print(f"Batch vid_nums: {vid_nums}")
    print(f"Unique vid_nums in batch: {len(set(vid_nums))}")
    assert len(set(vid_nums)) == batch_size, "Batch contains duplicate vid_nums!"
    
    # Test 4: Dataloader speed
    print("\n=== Test 4: Dataloader Speed ===")
    import time
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i == 10: break
    print(f"Time for 10 batches: {time.time() - start:.2f}s")
    
    print("\nAll tests passed! 🚀")