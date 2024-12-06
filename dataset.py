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
import torch.nn as nn
import torchaudio.transforms as T
from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings("ignore")

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=128, target_length=300):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.025 * sample_rate),    # 25ms window
            hop_length=int(0.010 * sample_rate),# 10ms hop
            n_mels=n_mels,
            power=2.0,
            normalized=True
        )
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process audio waveform to mel spectrogram"""
        #print("\n=== Audio Processing Debug ===")
        #print(f"1. Initial waveform - shape: {waveform.shape}, min: {waveform.min():.3f}, max: {waveform.max():.3f}")
        #print(f"   dtype: {waveform.dtype}, device: {waveform.device}")
        
        # Handle mono audio without channel dim
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        #print(f"2. After mono conversion - shape: {waveform.shape}, min: {waveform.min():.3f}, max: {waveform.max():.3f}")
            
        # Get mel spectrogram
        mel = self.mel_spec(waveform)  # (1, n_mels, time)
        #print(f"3. After mel spec - shape: {mel.shape}, min: {mel.min():.3f}, max: {mel.max():.3f}")
        #print(f"   Any NaNs?: {torch.isnan(mel).any()}")
        
        # Convert to db units and normalize
        mel = torch.log(mel + 1e-10)
        #print(f"4. After log - shape: {mel.shape}, min: {mel.min():.3f}, max: {mel.max():.3f}")
        #print(f"   Any NaNs?: {torch.isnan(mel).any()}")
        
        # Check mean and std before normalization
        mel_mean = mel.mean()
        mel_std = mel.std()
        #print(f"5. Before norm - mean: {mel_mean:.3f}, std: {mel_std:.3f}")
        
        mel = (mel - mel_mean) / (mel_std * 2)
        #print(f"6. After normalization - shape: {mel.shape}, min: {mel.min():.3f}, max: {mel.max():.3f}")
        #print(f"   Any NaNs?: {torch.isnan(mel).any()}")
        
        # Handle length
        current_length = mel.shape[2]
        if current_length > self.target_length:
            start = (current_length - self.target_length) // 2
            mel = mel[:, :, start:start + self.target_length]
        else:
            # For shorter segments, we repeat the audio
            repeats = (self.target_length + current_length - 1) // current_length
            mel = mel.repeat(1, 1, repeats)
            mel = mel[:, :, :self.target_length]
            
        #print(f"7. After length adjustment - shape: {mel.shape}, min: {mel.min():.3f}, max: {mel.max():.3f}")
        
        # Format for AST: (time, n_mels)
        mel = mel.squeeze(0).t()
        #print(f"8. Final output - shape: {mel.shape}, min: {mel.min():.3f}, max: {mel.max():.3f}")
        #print("============================\n")
        
        return mel

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
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 target_length: int = 300):
        
        self.video_paths = sorted([str(p) for p in Path(video_dir).glob("*.mp4")])
        self.frame_transform = frame_transform
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            target_length=target_length
        )
        
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
            #print(f"\nLoading audio from {video_path}")
            container = av.open(video_path)
            audio = container.streams.audio[0]
            
            audio_frames = []
            for frame in container.decode(audio=0):
                audio_frames.append(frame.to_ndarray())
            
            waveform = torch.from_numpy(np.concatenate(audio_frames))
            #print(f"Raw waveform - shape: {waveform.shape}, dtype: {waveform.dtype}")
            #print(f"Waveform stats - min: {waveform.min():.3f}, max: {waveform.max():.3f}, mean: {waveform.mean():.3f}")
            
            mel_spec = self.audio_processor(waveform)
            if torch.isnan(mel_spec).any():
                raise ValueError("NaN values found in mel_spec!")
            return mel_spec
            
        except Exception as e:
            print(f"Error loading audio from {video_path}: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_paths[idx]
        
        frame = self._load_video_frame(video_path)
        mel_spec = self._load_audio(video_path)
        
        return {
            'frame': frame,
            'mel_spec': mel_spec,
            'vid_num': self.vid_nums[idx]
        }

class ASTEmbedder(nn.Module):
    def __init__(self, 
                 fstride=128, 
                 tstride=2, 
                 input_fdim=128, 
                 input_tdim=300,
                 embed_dim=768):
        super().__init__()
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(1, embed_dim, 
                                   kernel_size=(input_fdim, tstride),
                                   stride=(fstride, tstride))
        
        # Calculate number of patches
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_out = self.patch_embed(test_input)
        num_patches = test_out.shape[2] * test_out.shape[3]
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        """
        Args:
            x: (batch_size, time_frame_num, frequency_bins)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        # Add channel dim and transpose
        # Print input stats
        #print(f"AST input stats - min: {x.min():.3f}, max: {x.max():.3f}, mean: {x.mean():.3f}")
        
        # Add channel dim and transpose
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = x.transpose(2, 3)  # (B, 1, F, T)
        
        # Print before patch embedding
        #print(f"Before patch embed - min: {x.min():.3f}, max: {x.max():.3f}, mean: {x.mean():.3f}")
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, E, P1, P2)
        x = x.flatten(2)  # (B, E, P)
        x = x.transpose(1, 2)  # (B, P, E)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        return x

if __name__ == "__main__":
    # Define transforms
    frame_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
    ])
    
    # Test the dataset
    dataset = AudioVisualDataset(
        video_dir='/home/cisco/heyo/densefuck/sound_of_pixels/dataset/solo_split_videos',
        frame_transform=frame_transform
    )
    
    # Test 1: Basic sample loading
    print("\n=== Test 1: Basic Sample Loading ===")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Frame shape: {sample['frame'].shape}")
    print(f"Mel spectrogram shape: {sample['mel_spec'].shape}")
    print(f"Video number: {sample['vid_num']}")
    
    # Test 2: Visualizations
    print("\n=== Test 2: Visualizations ===")
    plt.figure(figsize=(15, 5))
    
    # Show frame
    plt.subplot(1, 2, 1)
    frame = sample['frame'].permute(1, 2, 0).numpy()
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    plt.imshow(frame)
    plt.title(f"Frame from video {sample['vid_num']}")
    
    # Show mel spectrogram
    plt.subplot(1, 2, 2)
    plt.imshow(sample['mel_spec'].numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()
    
    # Test 3: Batch sampling
    print("\n=== Test 3: Batch Sampling ===")
    batch_size = 4  # Smaller batch size for testing
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
    
    # Test 4: Test AST Embedder
    print("\n=== Test 4: AST Embedder ===")
    ast_embedder = ASTEmbedder(
        input_fdim=128,  # Should match n_mels
        input_tdim=300,  # Should match target_length
        embed_dim=768
    )
    
    # Test with a batch from dataloader
    mel_specs = batch['mel_spec']  # (B, T, F)
    audio_embeddings = ast_embedder(mel_specs)
    print(f"Input mel_spec shape: {mel_specs.shape}")
    print(f"Output audio embeddings shape: {audio_embeddings.shape}")
    
    # Test 5: Dataloader speed
    print("\n=== Test 5: Dataloader Speed ===")
    import time
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i == 10: break
    print(f"Time for 10 batches: {time.time() - start:.2f}s")
    
    print("\nAll tests passed! 🚀")