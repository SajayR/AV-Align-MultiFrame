import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class TemporalVisualizer:
    def __init__(self, output_resolution=(1920, 1080)):
        self.output_res = output_resolution
        
        # Calculate frame sizes and positions
        self.frame_width = output_resolution[0] // 5   # 5 frames per row
        self.frame_height = (output_resolution[1] - 100) // 2  # 2 rows, leave space for progress
        self.positions = self._calculate_frame_positions()
        
        # Create custom colormap (transparent -> blue -> red -> yellow)
        colors = [
            (0,0,0,0),     # Transparent for low attention
            (0,0,1,0.5),   # Blue for medium-low
            (1,0,0,0.7),   # Red for medium-high  
            (1,1,0,1)      # Yellow for high attention
        ]
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)
        
    def _calculate_frame_positions(self):
        """Calculate positions for all 10 frames"""
        positions = []
        for row in range(2):
            for col in range(5):
                x = col * self.frame_width
                y = row * self.frame_height
                positions.append((x, y))
        return positions
    
    def _create_frame_overlay(self, frame: np.ndarray, heatmap: np.ndarray, alpha=0.5, 
                         border_size=3, padding=10):
        """Overlay heatmap on frame with border and padding"""
        # Calculate sizes accounting for padding and border
        inner_width = self.frame_width - 2 * (padding + border_size)
        inner_height = self.frame_height - 2 * (padding + border_size)
        
        # Create padded canvas (white background with black border)
        canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 30  # dark gray background
        
        # Draw border
        inner_start = (padding + border_size)
        cv2.rectangle(canvas,
                    (padding, padding),
                    (self.frame_width - padding, self.frame_height - padding),
                    (255, 255, 255),  # white border
                    border_size)
        
        # Normalize and enhance heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 0.5)  # Enhance contrast
        
        # Create colored heatmap
        heatmap_colored = self.cmap(heatmap)
        heatmap_bgr = (heatmap_colored[...,:3] * 255).astype(np.uint8)
        
        # Resize frame and heatmap to target size (inner area)
        frame_resized = cv2.resize(frame, (inner_width, inner_height))
        heatmap_resized = cv2.resize(heatmap_bgr, (inner_width, inner_height))
        
        # Blend frame and heatmap
        overlay = ((1-alpha) * frame_resized + alpha * heatmap_resized).astype(np.uint8)
        
        # Place the overlay in the padded canvas
        canvas[inner_start:inner_start + inner_height, 
            inner_start:inner_start + inner_width] = overlay
        
        return canvas
    
    def _draw_progress_bar(self, canvas: np.ndarray, progress: float):
        """Draw progress bar at bottom of canvas"""
        bar_height = 20
        bar_margin = 40
        bar_y = self.output_res[1] - bar_margin
        
        # Background bar
        cv2.rectangle(canvas, 
                     (bar_margin, bar_y), 
                     (self.output_res[0] - bar_margin, bar_y + bar_height),
                     (50, 50, 50),
                     -1)
        
        # Progress
        width = int((self.output_res[0] - 2*bar_margin) * progress)
        cv2.rectangle(canvas,
                     (bar_margin, bar_y),
                     (bar_margin + width, bar_y + bar_height),
                     (0, 255, 0),
                     -1)

    def visualize_dummy_data(self, num_frames=10, num_timesteps=50, output_path="test_viz.mp4"):
        """Test visualization with random data"""
        # Create dummy frames and attention maps
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                 for _ in range(num_frames)]
        
        attention_maps = [np.random.rand(num_frames, 224, 224) 
                         for _ in range(num_timesteps)]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            30,  # fps
            self.output_res
        )
        
        # Create frames
        # Create frames
        for t in range(num_timesteps):
            # Create blank canvas with dark background instead of black
            canvas = np.ones((self.output_res[1], self.output_res[0], 3), dtype=np.uint8) * 15  # very dark gray
            
            # Add each frame with its attention map
            for i, (x, y) in enumerate(self.positions):
                frame = frames[i]
                attn = attention_maps[t][i]
                overlay = self._create_frame_overlay(frame, attn)
                canvas[y:y+self.frame_height, x:x+self.frame_width] = overlay
            
            # Add progress bar
            self._draw_progress_bar(canvas, t / num_timesteps)
            
            writer.write(canvas)
        
        writer.release()

if __name__ == "__main__":
    # Test the visualizer with dummy data
    visualizer = TemporalVisualizer()
    visualizer.visualize_dummy_data()
    print("Done! Check test_viz.mp4")