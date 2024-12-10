import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import torch.nn as nn
import torchvision.transforms as transforms
from model import AudioVisualModel
from dataset import AudioVisualDataset, VideoBatchSampler
from viz import AudioVisualizer
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return mem_gb

def log_memory(step, location=""):
    mem_gb = get_memory_usage()
    print(f"Step {step} - {location} - Memory usage: {mem_gb:.2f} GB")

def memory_status(location=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    process = psutil.Process()
    ram = process.memory_info().rss / 1024**3
    print(f"\n=== Memory at {location} ===")
    print(f"RAM used: {ram:.2f} GB")
    print(f"CUDA Allocated: {allocated:.2f} GB")
    print(f"CUDA Reserved: {reserved:.2f} GB\n")

class AudioVisualTrainer:
    def __init__(
        self,
        video_dir: str,
        output_dir: str,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        num_workers: int = 12,
        vis_every: int = 250,
        num_vis_samples: int = 4,  # Number of videos to visualize
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_every = vis_every
        self.device = device
        self.use_wandb = use_wandb
        self.num_vis_samples = num_vis_samples

        # Setup logging
        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize dataset and dataloader
        frame_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

        self.dataset = AudioVisualDataset(
            video_dir=video_dir,
            frame_transform=frame_transform
        )

        self.batch_sampler = VideoBatchSampler(
            num_videos=len(self.dataset),
            batch_size=batch_size
        )

        #self.dataloader = DataLoader(
        #    self.dataset,
        #    batch_sampler=self.batch_sampler,
        #    num_workers=num_workers,
        #    #pin_memory=True
        #)
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            persistent_workers=True,  # Keep workers alive
            #prefetch_factor=2,  # Reduce prefetching
            worker_init_fn=lambda _: gc.collect()  # Cleanup on worker init
        )

        # Initialize model and optimizer
        self.model = AudioVisualModel().to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # Initialize visualizer
        self.visualizer = AudioVisualizer()

        # Save multiple random samples for visualization
        self.vis_samples = self._get_visualization_samples()

        if use_wandb:
            wandb.init(
                project="audio-visual-learning",
                config={
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                }
            )

    def _get_visualization_samples(self):
        """Get multiple random samples for visualization"""
        batch = next(iter(self.dataloader))
        indices = torch.randperm(len(batch['frame']))[:self.num_vis_samples]

        vis_samples = {
            'frames': batch['frame'][indices].to(self.device),
            'mel_specs': batch['mel_spec'][indices].to(self.device),
            'video_paths': [batch['video_path'][i] for i in indices]
        }

        return vis_samples

    def create_visualization(self, epoch: int, step: int):
        """Create visualizations for multiple samples with memory cleanup"""
        try:
            fig, axes = plt.subplots(self.num_vis_samples, 5, 
                                figsize=(20, 4*self.num_vis_samples))

            for i in range(self.num_vis_samples):
                # Create visualization
                self.visualizer.plot_attention_snapshot(
                    self.model,
                    self.vis_samples['frames'][i:i+1],
                    self.vis_samples['mel_specs'][i:i+1],
                    num_timesteps=5,
                    axes=axes[i] if self.num_vis_samples > 1 else axes
                )

                torch.cuda.empty_cache()  # Clear GPU memory after each sample

            if self.use_wandb:
                wandb.log({
                    "attention_snapshots": wandb.Image(plt),
                    "epoch": epoch,
                    "step": step
                })

            plt.close('all')  # Important: close all figures

            # Only save videos every few epochs to reduce memory pressure
            if epoch % 5 == 0:  # Adjust this number as needed
                for i in range(self.num_vis_samples):
                    video_path = self.output_dir / f'attentionepoch{epoch}sample{i}.mp4'

                    # Clear memory before video creation
                    torch.cuda.empty_cache()
                    gc.collect()

                    self.visualizer.make_attention_video(
                        self.model,
                        self.vis_samples['frames'][i:i+1],
                        self.vis_samples['mel_specs'][i:i+1],
                        video_path,
                        video_path=self.vis_samples['video_paths'][i]
                    )

                    if self.use_wandb:
                        wandb.log({
                            f"attentionvideo{i}": wandb.Video(str(video_path)),
                            "epoch": epoch,
                            "step": step
                        })

                    # Clear memory after each video
                    torch.cuda.empty_cache()
                    gc.collect()

        finally:
            plt.close('all')  # Ensure figures are closed even if there's an error
            torch.cuda.empty_cache()

    def train(self, num_epochs: int):
        step = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):

            self.model.train()
            epoch_losses = []

            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
            for batch in pbar:
                # Move batch to device
                self.model.train()
                frames = batch['frame'].to(self.device)
                audio = batch['mel_spec'].to(self.device)

                # Forward pass

                loss = self.model(frames, audio)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Log loss
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})

                # Clear some memory
                del frames, audio, loss
                torch.cuda.empty_cache()

                if step % 100 == 0:
                    gc.collect()  # Force garbage collection

                # Visualization with memory cleanup
                if step % self.vis_every == 0:

                    with torch.no_grad():
                        self.create_visualization(epoch, step)
                    plt.close('all')  # Close all matplotlib figures
                    gc.collect()

                step += 1

            # End of epoch

            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f'Epoch {epoch} - Loss: {epoch_loss:.4f}')

            if self.use_wandb:
                wandb.log({
                    'epoch_loss': epoch_loss,
                    'epoch': epoch,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })

            # Save checkpoint if best loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(epoch, step)

            # LR scheduler step
            self.scheduler.step()

            # Regular checkpoint every 10 epochs
            if epoch % 10 == 0:

                self.save_checkpoint(epoch, step)
                # Regular checkpoint every 10 epochs
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, step)

    def save_checkpoint(self, epoch: int, step: int):
        """Save a checkpoint of the model and training state"""
        checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}_step{step}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss if hasattr(self, 'best_loss') else float('inf'),
        }
        
        # Save with temp file to prevent corruption if interrupted
        temp_path = checkpoint_path.with_suffix('.temp.pt')
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        self.logger.info(f'Saved checkpoint to {checkpoint_path}')
        
        # Optionally log to wandb
        if self.use_wandb:
            wandb.save(str(checkpoint_path))

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and restore training state"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore best loss if it exists
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        return checkpoint['epoch'], checkpoint['step']
 


if __name__ == "__main__":
    trainer = AudioVisualTrainer(
        video_dir='/home/cisco/heyo/densefuck/VGGSound/vggsound_download/downloads',
        output_dir='./outputs',
        batch_size=8,
        num_epochs=500,
        learning_rate=1e-4,
        use_wandb=False  # Set to False if you don't want to use wandb
    )
    
    trainer.train(num_epochs=500)