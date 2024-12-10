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

class AudioVisualTrainer:
    def __init__(
        self,
        video_dir: str,
        output_dir: str,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        num_workers: int = 12,
        vis_every: int = 64,  # Visualize every N steps
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_every = vis_every
        self.device = device
        self.use_wandb = use_wandb
        
        # Setup logging
        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize dataset
        frame_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = AudioVisualDataset(
            video_dir=video_dir,
            frame_transform=frame_transform
        )
        
        # Initialize dataloader
        self.batch_sampler = VideoBatchSampler(
            self.dataset.vid_nums, 
            batch_size=batch_size
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=True
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
        
        # Save random sample for consistent visualization
        sample = next(iter(self.dataloader))
        self.vis_frame = sample['frame'][0:1].to(device)
        self.vis_audio = sample['mel_spec'][0:1].to(device)
        self.original_video_path = sample['video_path'][0]
        
        if use_wandb:
            wandb.init(
                project="audio-visual-learning",
                config={
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                }
            )
    
    def save_checkpoint(self, epoch: int, step: int):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        torch.save(
            checkpoint,
            self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        )
    
    def create_visualization(self, epoch: int, step: int):
        # Create snapshot plot
        self.visualizer.plot_attention_snapshot(
            self.model, self.vis_frame, self.vis_audio,
            num_timesteps=5
        )
        
        if self.use_wandb:
            wandb.log({
                "attention_snapshots": wandb.Image(plt),
                "epoch": epoch,
                "step": step
            })
        
        plt.close()
        
        # Save video every epoch
        if epoch % 1 == 0:
            video_path = self.output_dir / f'attention_epoch_{epoch}.mp4'
            
            self.visualizer.make_attention_video(
                self.model, 
                self.vis_frame, 
                self.vis_audio,
                video_path,
                video_path=self.original_video_path
            )
            
            if self.use_wandb:
                wandb.log({
                    "attention_video": wandb.Video(str(video_path)),
                    "epoch": epoch,
                    "step": step
                })
    
    def train(self, num_epochs: int):
        step = 0
        best_loss = float('inf')
        # Load weights if a checkpoint is provided
        
        for epoch in range(num_epochs):
            self.model.train()
            #for name, param in self.model.named_parameters():
                #print(f"Parameter: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
            epoch_losses = []
            
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
            for batch in pbar:
                self.model.train()
                # Move batch to device
                frames = batch['frame'].to(self.device)
                audio = batch['mel_spec'].to(self.device)
                
                # Forward pass
                # Check for NaNs in frames and audio
                if torch.isnan(frames).any():
                    raise ValueError("NaN values found in frames!")
                if torch.isnan(audio).any():
                    raise ValueError("NaN values found in audio!")
                loss = self.model(frames, audio)
                #print(f"Loss training: {loss}")
                # Backward pass
                self.optimizer.zero_grad()
                
                # Check weights before backward pass on first iteration
                if epoch == 0 and step == 0:
                    weights_before = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                
                loss.backward()
                
                # Check if weights were updated on first iteration
                if epoch == 0 and step == 0:
                    weights_after = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                    for name in weights_before:
                        if not torch.equal(weights_before[name], weights_after[name]):
                            print(f"Weights updated for {name}")
                        else:
                            print(f"Warning: Weights not updated for {name}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Log loss
                loss_value = loss.item()
                if torch.isnan(loss):
                    raise ValueError("NaN values found in loss!")
                epoch_losses.append(loss_value)
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                
                if self.use_wandb:
                    wandb.log({
                        'step_loss': loss_value,
                        'epoch': epoch,
                        'step': step
                    })
                
                # Visualization
                if step % self.vis_every == 0:
                    self.create_visualization(epoch, step)
                
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
            #if epoch_loss < best_loss:
                #best_loss = epoch_loss
                #self.save_checkpoint(epoch, step)
            
            # LR scheduler step
            self.scheduler.step()
            
            # Regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, step)

if __name__ == "__main__":
    trainer = AudioVisualTrainer(
        video_dir='/home/cisco/heyo/densefuck/sound_of_pixels/dataset/solo_split_videos',
        output_dir='./outputs',
        batch_size=32,
        num_epochs=500,
        learning_rate=1e-4,
        use_wandb=False  # Set to False if you don't want to use wandb
    )
    
    trainer.train(num_epochs=500)