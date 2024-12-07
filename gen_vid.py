import torch
import torchvision.transforms as transforms
from pathlib import Path
import argparse

from model import AudioVisualModel
from dataset import AudioVisualDataset
from viz import AudioVisualizer

def generate_attention_videos(
    checkpoint_path: str,
    video_dir: str,
    output_dir: str,
    num_videos: int = 5,
    device: str = 'cuda'
):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and load checkpoint
    model = AudioVisualModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Initialize dataset
    frame_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AudioVisualDataset(
        video_dir=video_dir,
        frame_transform=frame_transform
    )
    
    # Initialize visualizer
    visualizer = AudioVisualizer()
    
    # Generate videos for random samples
    import random
    indices = random.sample(range(len(dataset)), num_videos)
    
    for i, idx in enumerate(indices):
        print(f"\nProcessing video {i+1}/{num_videos}")
        sample = dataset[idx]
        
        # Move tensors to device
        frame = sample['frame'].unsqueeze(0).to(device)
        audio = sample['mel_spec'].unsqueeze(0).to(device)
        
        # Generate attention visualization
        video_path = output_dir / f'attention_video_{i}.mp4'
        print(f"Generating video: {video_path}")
        visualizer.make_attention_video(
            model, frame, audio,
            str(video_path)
        )
        
        # Generate snapshot
        snapshot_path = output_dir / f'attention_snapshot_{i}.png'
        print(f"Generating snapshot: {snapshot_path}")
        visualizer.plot_attention_snapshot(
            model, frame, audio,
            num_timesteps=5
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate attention visualization videos')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output videos')
    parser.add_argument('--num_videos', type=int, default=5,
                        help='Number of videos to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    generate_attention_videos(
        checkpoint_path=args.checkpoint,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        num_videos=args.num_videos,
        device=args.device
    )