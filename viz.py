import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class AudioVisualizer:
    def __init__(self, patch_size=16, image_size=224):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size
        
        # Create a custom colormap (transparent -> red)
        colors = [(1,0,0,0), (1,0,0,1)]  # R,G,B,A
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)
        
    def get_attention_maps(self, model, frame, audio):
        """
        Get attention maps for each audio token
        
        Args:
            model: AudioVisualModel instance
            frame: (1, C, H, W) single frame tensor
            audio: (1, T, F) audio spectrogram tensor
            
        Returns:
            attention_maps: (Na, H, W) attention score for each audio token
        """
        model.eval()
        with torch.no_grad():
            # Get embeddings
            visual_feats = model.visual_embedder(frame)     # (1, Nv, D)
            audio_feats = model.audio_embedder(audio)       # (1, Na, D)
            
            # Compute token-level similarities
            similarity = model.compute_similarity_matrix(
                audio_feats, 
                visual_feats
            ).squeeze(0)  # (Na, Nv)
            
            # Convert patch attention to pixel attention
            attention_maps = self.patches_to_heatmaps(similarity)
            
        return attention_maps
    
    def patches_to_heatmaps(self, patch_attention):
        """
        Convert patch-level attention to pixel-level heatmaps
        
        Args:
            patch_attention: (Na, Nv) attention scores for each audio token
            
        Returns:
            heatmaps: (Na, H, W) upsampled attention maps
        """
        Na, Nv = patch_attention.shape
        
        # Reshape to square grid
        patches = patch_attention.reshape(Na, self.num_patches, self.num_patches)
        
        # Upsample to image size
        heatmaps = F.interpolate(
            patches.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        #print("heatmaps.shape", heatmaps.shape)
        #print("Is first equal to second?", (heatmaps[0] == heatmaps[1]).all())
        #print("First heatmap:", heatmaps[0])
        #print("Second heatmap:", heatmaps[1])
        return heatmaps
    
    def create_overlay_frame(self, frame: np.ndarray, heatmap: np.ndarray, alpha=0.6):
        """
        Create a single frame with heatmap overlay
        
        Args:
            frame: (H, W, C) original frame
            heatmap: (H, W) attention heatmap
            alpha: transparency of the overlay
            
        Returns:
            overlay: (H, W, C) frame with heatmap overlay
        """
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create RGBA heatmap
        heatmap_colored = self.cmap(heatmap)
        
        # Convert to BGR for cv2
        heatmap_bgr = (heatmap_colored[...,:3] * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(frame, 1.0, heatmap_bgr, alpha, 0)
        
        return overlay
    
    def make_attention_video(self, model, frame, audio, output_path, fps=30):
        """
        Create a video showing attention overlay for each audio token
        
        Args:
            model: AudioVisualModel instance
            frame: (1, C, H, W) single frame tensor
            audio: (1, T, F) audio spectrogram tensor
            output_path: path to save the video
            fps: frames per second for the output video
        """
        # Get attention maps
        attention_maps = self.get_attention_maps(model, frame, audio)
        
        # Debug: Print attention map statistics
        #print("Attention maps shape:", attention_maps.shape)
        #print("Attention maps min:", attention_maps.min().item())
        #print("Attention maps max:", attention_maps.max().item())
        
        # Convert frame to numpy
        frame_np = (frame.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        print("frame_np.shape", frame_np.shape)
        print("frame_np.dtype", frame_np.dtype)
        print("frame_np.max()", frame_np.max())
        print("frame_np.min()", frame_np.min())
        
        # Setup video writer
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.image_size, self.image_size)
        )
        frame_count = 0
        # Create each frame
        for i, heatmap in enumerate(attention_maps.cpu().numpy()):
            # Debug: Print each heatmap's min and max
            #print(f"Heatmap {i} min:", heatmap.min(), "max:", heatmap.max())
            
            overlay = self.create_overlay_frame(frame_np, heatmap)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            frame_count += 1
        print("Frame count:", frame_count)    
        writer.release()
        
    def plot_attention_snapshot(self, model, frame, audio, num_timesteps=5):
        """
        Plot a grid of attention maps at different timesteps
        
        Args:
            model: AudioVisualModel instance
            frame: (1, C, H, W) single frame tensor
            audio: (1, T, F) audio spectrogram tensor
            num_timesteps: number of timesteps to show
        """
        attention_maps = self.get_attention_maps(model, frame, audio)
        frame_np = (frame.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        
        # Select evenly spaced timesteps
        timesteps = np.linspace(0, len(attention_maps)-1, num_timesteps).astype(int)
        
        # Create plot
        fig, axes = plt.subplots(1, num_timesteps, figsize=(4*num_timesteps, 4))
        if num_timesteps == 1:
            axes = [axes]
            
        for ax, t in zip(axes, timesteps):
            heatmap = attention_maps[t].cpu().numpy()
            overlay = self.create_overlay_frame(frame_np, heatmap)
            ax.imshow(overlay)
            ax.set_title(f'Time: {t/50:.2f}s')  # Assuming 50 tokens per second
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test visualization
    from model import AudioVisualModel
    
    model = AudioVisualModel()
    visualizer = AudioVisualizer()
    
    # Create dummy data
    frame = torch.randn(1, 3, 224, 224)
    audio = torch.randn(1, 300, 128)
    
    # Test video creation
    visualizer.make_attention_video(
        model, frame, audio,
        'test_attention.mp4'
    )
    
    # Test snapshot visualization
    visualizer.plot_attention_snapshot(
        model, frame, audio,
        num_timesteps=5
    )