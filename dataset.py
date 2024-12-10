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
from tqdm import tqdm
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
        # Handle mono audio without channel dim
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Get mel spectrogram
        mel = self.mel_spec(waveform)  # (1, n_mels, time)
        
        # Convert to db units and normalize
        mel = torch.log(mel + 1e-10)
        # Check mean and std before normalization
        mel_mean = mel.mean()
        mel_std = mel.std()

        
        mel = (mel - mel_mean) / (mel_std * 2)
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
 
        # Format for AST: (time, n_mels)
        mel = mel.squeeze(0).t()
        return mel

class AudioProcessor:
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=128,
                 target_length=998,  # Calculated for 10s clips
                 ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        
        # Window and hop length for 25ms windows with 10ms hop
        self.n_fft = int(0.025 * sample_rate)
        self.hop_length = int(0.010 * sample_rate)
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=True,
            center=True,
            pad_mode='reflect',
            norm='slaney',  # Using slaney norm as it's standard for audio processing
            mel_scale='htk'  # HTK scale is standard for audio classification
        )
        
            
    def load_audio(self, audio_path: str, start_time: float = 0.0) -> torch.Tensor:
        """Load audio file and extract 10s segment"""
        waveform, sr = torchaudio.load(audio_path, normalize=True)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Extract 10s segment
        start_sample = int(start_time * self.sample_rate)
        segment_samples = 10 * self.sample_rate
        
        if waveform.shape[1] < segment_samples:
            # Pad if audio is too short
            pad_length = segment_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:, start_sample:start_sample + segment_samples]
            
        return waveform
            
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process audio waveform to mel spectrogram"""
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Get mel spectrogram
        mel = self.mel_spec(waveform)  # (1, n_mels, time)
        
        # Convert to db units and normalize
        mel = torch.log(mel + 1e-10)
        mel = (mel - mel.mean()) / (mel.std() + 1e-10)
        
        # Handle length
        current_length = mel.shape[2]
        if current_length > self.target_length:
            start = (current_length - self.target_length) // 2
            mel = mel[:, :, start:start + self.target_length]
        else:
            # For shorter segments, we pad with repetition
            repeats = (self.target_length + current_length - 1) // current_length
            mel = mel.repeat(1, 1, repeats)
            mel = mel[:, :, :self.target_length]
        
        # Format for AST: (time, n_mels)
        mel = mel.squeeze(0).t()
        return mel

class VideoBatchSampler(Sampler):
    """Simple random batch sampler"""
    def __init__(self, num_videos: int, batch_size: int):
        self.num_videos = num_videos
        self.batch_size = batch_size
        
    def __iter__(self):
        # Create list of indices and shuffle
        indices = list(range(self.num_videos))
        random.shuffle(indices)
        
        # Yield batches
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            yield indices[i:i + self.batch_size]
    
    def __len__(self):
        return self.num_videos // self.batch_size

class AudioVisualDataset(Dataset):
    def __init__(self, 
                 video_dir: str,
                 frame_transform=None,
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 target_length: int = 998):
        
        self.frame_transform = frame_transform
        self.video_paths = []
        
        # Find all video files
        all_videos = sorted([str(p) for p in Path(video_dir).glob("*.mp4")])
        print(f"Found {len(all_videos)} video files")
        
        # Validate videos during initialization
        print("Validating videos...")
        for video_path in tqdm(all_videos):
            try:
                # Try to open the video and check if it has both video and audio streams
                container = av.open(video_path)
                if container.streams.video and container.streams.audio:
                    self.video_paths.append(video_path)
                container.close()
            except Exception as e:
                print(f"Skipping corrupted video {video_path}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(self.video_paths)} valid videos")
        
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            target_length=target_length
        )
    
    def _load_video_frame(self, video_path: str) -> torch.Tensor:
        """Load middle frame from video with error handling and proper resource cleanup"""
        max_retries = 3
        container = None
        
        for attempt in range(max_retries):
            try:
                container = av.open(video_path)
                if not container.streams.video:
                    raise ValueError("No video stream found")
                
                video_stream = container.streams.video[0]
                video_stream.thread_type = "AUTO"
                
                # Get middle frame
                total_frames = video_stream.frames
                if total_frames == 0:
                    raise ValueError("Video has 0 frames")
                    
                target_frame = total_frames // 2
                
                for i, frame in enumerate(container.decode(video=0)):
                    if i == target_frame:
                        numpy_frame = frame.to_ndarray(format='rgb24')
                        frame_tensor = torch.from_numpy(numpy_frame).permute(2, 0, 1).float() / 255.0
                        
                        if self.frame_transform:
                            frame_tensor = self.frame_transform(frame_tensor)
                            
                        return frame_tensor
                        
                raise ValueError("Could not reach target frame")
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Failed to load video after {max_retries} attempts: {video_path}")
                    raise e
                time.sleep(0.1)  # Small delay before retry
                
            finally:
                # Ensure container is always closed, even if an error occurs
                if container:
                    try:
                        container.close()
                    except:
                        pass
            
    def _load_audio(self, video_path: str) -> torch.Tensor:
        """Load and process audio with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                container = av.open(video_path)
                if not container.streams.audio:
                    raise ValueError("No audio stream found")
                
                audio_stream = container.streams.audio[0]
                
                audio_frames = []
                for frame in container.decode(audio=0):
                    audio_frames.append(frame.to_ndarray())
                
                if not audio_frames:
                    raise ValueError("No audio frames decoded")
                    
                waveform = torch.from_numpy(np.concatenate(audio_frames))
                mel_spec = self.audio_processor(waveform)
                
                if torch.isnan(mel_spec).any():
                    raise ValueError("NaN values found in mel_spec")
                    
                container.close()
                return mel_spec
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Failed to load audio after {max_retries} attempts: {video_path}")
                    raise e
                time.sleep(0.1)  # Small delay before retry
                
            finally:
                try:
                    container.close()
                except:
                    pass
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_paths[idx]
        
        try:
            frame = self._load_video_frame(video_path)
            mel_spec = self._load_audio(video_path)
            
            return {
                'frame': frame,
                'mel_spec': mel_spec,
                'video_path': video_path
            }
            
        except Exception as e:
            print(f"Error loading sample {video_path}: {str(e)}")
            # Return a different valid sample
            new_idx = (idx + 1) % len(self)
            return self[new_idx]

class PatchEmbed(nn.Module):
    """
    2D patch embedding layer specifically designed for audio spectrograms
    """
    def __init__(self, 
                 input_fdim=128,      # Frequency dimension (mel bins)
                 input_tdim=998,      # Time dimension (for 10s audio)
                 patch_size=(16, 16), # Patch size in (freq, time)
                 embed_dim=768):      # Embedding dimension
        super().__init__()
        
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.num_patches = (input_fdim // patch_size[0]) * (input_tdim // patch_size[1])
        
        # Convolutional layer for patch embedding
        self.proj = nn.Conv2d(1,                      # Input channels (1 for mel spec)
                             embed_dim,               # Output embedding dimension
                             kernel_size=patch_size,  
                             stride=patch_size)       # Non-overlapping patches
    
    def forward(self, x):
        # x shape: (B, T, F) -> need (B, 1, F, T)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        
        # Project to embedding space
        x = self.proj(x)  # (B, embed_dim, H, W)
        
        # Flatten patches and transpose
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x

class ASTEmbedder(nn.Module):
    def __init__(self,
                 input_fdim=128,     # Number of mel bins
                 input_tdim=998,     # Time frames for 10s
                 patch_size=(16, 16),# Patch size
                 embed_dim=768,      # Embedding dimension
                 depth=12,           # Number of transformer layers
                 num_heads=12,       # Number of attention heads
                 mlp_ratio=4.,       # MLP hidden dim ratio
                 drop_rate=0.1,      # Dropout rate
                 attn_drop_rate=0.1):# Attention dropout rate
        super().__init__()
        
        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            
        # Initialize position embeddings
        if isinstance(m, ASTEmbedder):
            nn.init.trunc_normal_(m.pos_embed, std=0.02)
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, time_frames, frequency_bins)
                For 10s audio: (B, 998, 128)
        Returns:
            encoded: (batch_size, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
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
        frame_transform=frame_transform,
        num_frames=1
    )
    
    # Test 1: Basic sample loading
    print("\n=== Test 1: Basic Sample Loading ===")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Frame shape: {sample['frames'].shape}")
    print(f"Mel spectrogram shape: {sample['mel_spec'].shape}")
    print(f"Video number: {sample['vid_num']}")
    
    # Test 2: Visualizations
    print("\n=== Test 2: Visualizations ===")
    plt.figure(figsize=(15, 5))
    
    # Show frame
    plt.subplot(1, 2, 1)
    frame = sample['frames'].permute(1, 2, 0).numpy()
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
        input_tdim=998,  # Should match target_length
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
    processor = AudioProcessor()
    dummy_audio = torch.randn(1, 160000)  # 10s at 16kHz
    
    # Process audio
    mel_spec = processor(dummy_audio)
    print(f"Mel spectrogram shape: {mel_spec.shape}")  # Should be (998, 128)
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()
    plt.show()

    batch_size = 4
    model = ASTEmbedder()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Create dummy input
    x = torch.randn(batch_size, 998, 128)
    
    # Forward pass
    with torch.no_grad():
        out = model(x)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")