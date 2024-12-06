# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
#os.environ['TORCH_HOME'] = '../../pretrained_models'
#import timm
from timm.models.layers import to_2tuple,trunc_normal_

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

class AudioPreprocessor:
    def __init__(self, 
                 sample_rate=16000,  # Standard sample rate
                 n_mels=128,         # Number of mel bins
                 target_length=1024): # Target time frames
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        
        # Initialize mel spectrogram converter
        # Window of 25ms = 0.025 * sample_rate
        # Hop of 10ms = 0.010 * sample_rate
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.010 * sample_rate),  # 10ms hop
            n_mels=n_mels,
            power=2.0,
            normalized=True
        )

        # For resampling if needed
        self.resampler = None
    
    def load_audio(self, file_path):
        """Load audio file and resample if needed"""
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            if self.resampler is None:
                self.resampler = T.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)
            
        return waveform

    def process(self, file_path):
        """Complete preprocessing pipeline"""
        # Load and normalize audio
        waveform = self.load_audio(file_path)
        
        # Get mel spectrogram
        mel = self.mel_spec(waveform)  # (1, n_mels, time)
        
        # Convert to db units
        mel = torch.log(mel + 1e-10)
        
        # Normalize to zero mean and unit variance
        # The paper mentions normalizing to mean=0, std=0.5
        mel = (mel - mel.mean()) / (mel.std() * 2)
        
        # Handle length
        current_length = mel.shape[2]
        if current_length > self.target_length:
            # Center crop if too long
            start = (current_length - self.target_length) // 2
            mel = mel[:, :, start:start + self.target_length]
        elif current_length < self.target_length:
            # Pad if too short
            pad_length = self.target_length - current_length
            mel = F.pad(mel, (0, pad_length), mode='constant')
            
        # Remove channel dim and transpose to match AST input shape
        # From (1, n_mels, time) to (time, n_mels)
        mel = mel.squeeze(0).t()
        
        return mel

    @torch.no_grad()
    def process_batch(self, file_paths):
        """Process a batch of files"""
        mels = [self.process(f) for f in file_paths]
        return torch.stack(mels)

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTPatchEmbedder(nn.Module):
    def __init__(self, fstride=128, tstride=2, input_fdim=128, input_tdim=1024, 
                 embed_dim=768, num_heads=12, num_layers=12):
        super(ASTPatchEmbedder, self).__init__()
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(1, embed_dim, 
                                   kernel_size=(128, 2), 
                                   stride=(fstride, tstride))
        
        # Calculate number of patches
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        self.num_patches = f_dim * t_dim
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_out = self.patch_embed(test_input)
        return test_out.shape[2], test_out.shape[3]

    def forward(self, x):
        """
        Args:
            x: (batch_size, time_frame_num, frequency_bins)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        # Add channel dim and transpose
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = x.transpose(2, 3)  # (B, 1, F, T)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, E, P1, P2)
        x = x.flatten(2)  # (B, E, P)
        x = x.transpose(1, 2)  # (B, P, E)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transform
        x = self.transformer(x)
        x = self.norm(x)
        
        return x
    

class AudioPreprocessor:
    def __init__(self, 
                 sample_rate=16000,  # Standard sample rate
                 n_mels=128,         # Number of mel bins
                 target_length=300): # Target time frames
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        
        # Initialize mel spectrogram converter
        # Window of 25ms = 0.025 * sample_rate
        # Hop of 10ms = 0.010 * sample_rate
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.010 * sample_rate),  # 10ms hop
            n_mels=n_mels,
            power=2.0,
            normalized=True
        )

    def load_audio(self, file_path):
        """Load audio from video file and resample if needed"""
        # Use librosa.load to load audio
        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        waveform = torch.from_numpy(waveform).unsqueeze(0)  # Add channel dimension
        return waveform

    def process(self, file_path):
        """Complete preprocessing pipeline"""
        # Load and normalize audio
        waveform = self.load_audio(file_path)
        
        # Get mel spectrogram
        mel = self.mel_spec(waveform)  # (1, n_mels, time)
        
        # Convert to db units
        mel = torch.log(mel + 1e-10)
        
        # Normalize to zero mean and unit variance
        mel = (mel - mel.mean()) / (mel.std() * 2)
        
        # Handle length
        current_length = mel.shape[2]
        if current_length > self.target_length:
            # Center crop if too long
            start = (current_length - self.target_length) // 2
            mel = mel[:, :, start:start + self.target_length]
        elif current_length < self.target_length:
            # Pad if too short
            pad_length = self.target_length - current_length
            mel = F.pad(mel, (0, pad_length), mode='constant')
                
        # Remove channel dim and transpose to match AST input shape
        # From (1, n_mels, time) to (time, n_mels)
        mel = mel.squeeze(0).t()
        
        return mel

    @torch.no_grad()
    def process_batch(self, file_paths):
        """Process a batch of files"""
        mels = [self.process(f) for f in file_paths]
        return torch.stack(mels)


def extract_audio_features_from_videos(video_paths):
    """Extract audio features from a list of video paths and return batched features."""
    # Initialize the preprocessor with target_length=300
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=128,
        target_length=300
    )

    # Initialize the patch embedder with input_tdim matching the target_length
    patch_embedder = ASTPatchEmbedder(
        fstride=128,
        tstride=2,
        input_fdim=128,
        input_tdim=300  # Should match target_length
    )

    # Process the batch of videos
    mel_specs = preprocessor.process_batch(video_paths)
    embeddings = patch_embedder(mel_specs)
    return embeddings[:, :128, :]







if __name__ == '__main__':
    video_paths = [
        "/home/cisco/heyo/densefuck/sound_of_pixels/dataset/solo_split_videos/0_0.mp4",
        "/home/cisco/heyo/densefuck/sound_of_pixels/dataset/solo_split_videos/0_1.mp4",
        # Add more video paths as needed
    ]
    embeddings = extract_audio_features_from_videos(video_paths)
    print("Batch embeddings shape:", embeddings.shape)  #batch size, 128 time frames, 768 embedding dim 




