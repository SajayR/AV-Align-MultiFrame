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
import warnings
warnings.filterwarnings("ignore")
import time

def collate_fn(batch):
    # Get all tokens (already processed)
    video_tokens = torch.stack([item['video_frames'] for item in batch])
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    audio_padded = torch.zeros(len(batch), max_audio_len)
    for i, item in enumerate(batch):
        audio_len = item['audio'].shape[0]
        audio_padded[i, :audio_len] = item['audio']
    
    return {
        'frame': video_tokens,
        'audio': audio_padded,
        'vid_nums': [item['vid_num'] for item in batch],
        'segment_nums': [item['segment_num'] for item in batch],
        'video_paths': [str(item['video_path']) for item in batch]  # Convert PosixPath to string
    }

class AudioVisualTrainer:
    def __init__(
        self,
        video_dir: str,
        output_dir: str,
        batch_size: int = 32,
        num_epochs: int = 400,
        learning_rate: float = 1e-3,
        num_workers: int = 12,
        vis_every: int = 1000,
        num_vis_samples: int = 10,
        device: str = 'cuda',
        use_wandb: bool = False,
        force_new_training: bool = False,
        gradient_accumulation_steps: int = 1,
        unfreeze_hubert_epoch: int = 10,
        unfreeze_vit_epoch: int = 20,
        save_every_steps: int = 3000
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_every = vis_every
        self.device = device
        self.use_wandb = use_wandb
        self.num_vis_samples = num_vis_samples
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model = AudioVisualModel().to(device)
        self.save_every_steps = save_every_steps
        
        self.config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'vis_every': vis_every,
            'num_vis_samples': num_vis_samples,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'unfreeze_hubert_epoch': unfreeze_hubert_epoch,
            'unfreeze_vit_epoch': unfreeze_vit_epoch
        }

        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        self.dataset = AudioVisualDataset(
            data_root=video_dir,
            sample_fps=20
        )

        self.batch_sampler = VideoBatchSampler(
            vid_nums=self.dataset.vid_nums,
            batch_size=self.config['batch_size']
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.config['num_workers'],
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=8
        )
        
        num_training_steps = len(self.dataloader) * self.config['num_epochs']

        # Initially freeze Vision and HuBERT parameters
        for param in self.model.visual_embedder.model.parameters():
            param.requires_grad = False
        for param in self.model.audio_embedder.hubert.parameters():
            param.requires_grad = False

        # Identify parameter groups once and for all
        projection_params = []
        temperature_params = []
        hubert_params = []
        vit_params = []

        for name, param in self.model.named_parameters():
            if "audio_embedder.hubert" in name:
                hubert_params.append(param)
            elif "visual_embedder.model" in name:
                vit_params.append(param)
            elif "projection" in name:
                projection_params.append(param)
            elif "temperature" in name:
                temperature_params.append(param)
            else:
                projection_params.append(param)

        # We include all parameters from the start.
        # Huberts and VIT params are initially requires_grad=False, so no updates occur.
        self.optimizer = torch.optim.AdamW(
            [
                {'params': projection_params, 'lr': 1e-3},
                {'params': temperature_params, 'lr': 1e-3},
                {'params': hubert_params, 'lr': 1e-5},
                {'params': vit_params, 'lr': 1e-5}
            ]
        )
        
        # Using OneCycleLR or another scheduler that depends on total steps is fine.
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=num_training_steps,
            pct_start=0.01,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

        self.visualizer = AudioVisualizer()

        if use_wandb:
            if not force_new_training and self.find_latest_checkpoint():
                pass
            else:
                wandb.init(
                    project="DenseGod",
                    name="DenseFuck",
                    config=self.config
                )

        self.vis_samples = self._get_visualization_samples()

        if not force_new_training:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                print(f"Found checkpoint: {checkpoint_path}")
                self.load_checkpoint(checkpoint_path)

        if wandb.run is None:
            print("No wandb run found, initializing new run")
            wandb.init(
                project="DenseGod",
                name="DenseFuck",
                config=self.config
            )

    def find_latest_checkpoint(self):
        checkpoints = list(self.output_dir.glob('checkpoint_epoch*.pt'))
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: (
            int(str(x).split('epoch')[1].split('_')[0]),
            int(str(x).split('step')[1].split('.')[0])
        ))
        return latest

    def save_checkpoint(self, epoch: int, step: int):
        checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}_step{step}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'vis_samples': {
                'frames': self.vis_samples['frames'].cpu(),
                'audios': self.vis_samples['audios'].cpu(),
                'video_paths': self.vis_samples['video_paths']
            }
        }
        if self.use_wandb and wandb.run is not None:
            checkpoint['wandb_run_id'] = wandb.run.id
        temp_path = checkpoint_path.with_suffix('.temp.pt')
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        self.logger.info(f'Saved checkpoint to {checkpoint_path}')
        print(f"Saved checkpoint for epoch {epoch} and step {step}.")

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        '''self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.config.update(checkpoint.get('config', {}))

        # No need to rebuild parameter groups since we have them from the start
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])'''

        if 'vis_samples' in checkpoint:
            self.vis_samples = {
                'frames': checkpoint['vis_samples']['frames'].to(self.device),
                'audios': checkpoint['vis_samples']['audios'].to(self.device),
                'video_paths': checkpoint['vis_samples']['video_paths']
            }
        
        '''if self.use_wandb:
            wandb_run_id = checkpoint.get('wandb_run_id')
            if wandb_run_id is not None:
                wandb.init(
                    project="DenseGod",
                    id=wandb_run_id,
                    resume="must"
                )
            else:
                wandb.init(
                    project="DenseGod",
                    name=f"DenseFuck",
                    config=self.config
                )

        print(f"Resumed from epoch {self.start_epoch} (step {self.global_step})")
        print(f"Best loss so far: {self.best_loss:.4f}")'''

    def _get_visualization_samples(self):
        batch = next(iter(self.dataloader))
        indices = torch.randperm(len(batch['frame']))[:self.num_vis_samples]
        vis_samples = {
            'frames': batch['frame'][indices].to(self.device),
            'audios': batch['audio'][indices].to(self.device),
            'video_paths': [batch['video_paths'][i] for i in indices]
        }
        return vis_samples

    def create_visualization(self, epoch: int, step: int):
        try:
            fig, axes = plt.subplots(self.num_vis_samples, 5, figsize=(20, 4*self.num_vis_samples))
            for i in range(self.num_vis_samples):
                self.visualizer.plot_attention_snapshot(
                    self.model,
                    self.vis_samples['frames'][i:i+1],
                    self.vis_samples['audios'][i:i+1],
                    num_timesteps=5,
                    axes=axes[i] if self.num_vis_samples > 1 else axes
                )
                if self.use_wandb:
                    wandb.log({
                        "attention_snapshots": wandb.Image(plt),
                        "epoch": epoch,
                        "step": step
                    })
                torch.cuda.empty_cache()
            plt.close('all')

            if epoch % 1 == 0:
                print(f"Saving attention videos for epoch {epoch}")
                for i in range(self.num_vis_samples):
                    video_path = self.output_dir / f'attention_epoch{epoch}_sample{i}.mp4'
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.visualizer.make_attention_video(
                        self.model,
                        self.vis_samples['frames'][i:i+1],
                        self.vis_samples['audios'][i:i+1],
                        video_path,
                        video_path=self.vis_samples['video_paths'][i]
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

        finally:
            plt.close('all')
            torch.cuda.empty_cache()

    def train(self, num_epochs: int = None):
        if num_epochs is not None:
            self.config['num_epochs'] = num_epochs
            # If desired, a different scheduler could be reinitialized here.
            # But ideally, stick with the initial one.

        accumulation_counter = 0
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"Epoch {epoch}")
            #print("Unfreezing HuBERT at epoch", self.config['unfreeze_hubert_epoch'])
            #print("Unfreezing ViT at epoch", self.config['unfreeze_vit_epoch'])
            
            # Unfreeze logic: Just set requires_grad = True without re-init optimizer
            if epoch == self.config['unfreeze_hubert_epoch']:
                print(f"Unfreezing HuBERT parameters at epoch {epoch}")
                for param in self.model.audio_embedder.hubert.parameters():
                    param.requires_grad = True

            if epoch == self.config['unfreeze_vit_epoch']:
                print(f"Unfreezing ViT parameters at epoch {epoch}")
                for param in self.model.visual_embedder.model.parameters():
                    param.requires_grad = True

            self.model.train()
            epoch_losses = []
            print("Training the following layers:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")

            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
            for batch in pbar:
                self.model.train()
                frames = batch['frame'].to(self.device)
                audio = batch['audio'].to(self.device)
                loss = self.model(frames, audio)

                if loss.item() > 10:
                    print(f"Skipping batch with loss: {loss.item():.4f}")
                    continue

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                accumulation_counter += 1

                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    if self.global_step % 10000 == 0:
                        print("\nGradient Analysis:")
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                param_norm = param.data.norm().item()
                                print(f"{name[:30]:30} | grad_norm: {grad_norm:10.4f} | param_norm: {param_norm:10.4f}")
                        print("\n")

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss.item() * self.gradient_accumulation_steps,
                        "temperature": self.model.temperature.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })

                loss_value = loss.item() * self.gradient_accumulation_steps
                epoch_losses.append(loss_value)
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})

                del frames, audio, loss
                torch.cuda.empty_cache()

                if self.global_step % 100 == 0:
                    gc.collect()

                if self.global_step % self.vis_every == 0:
                    with torch.no_grad():
                        self.create_visualization(epoch, self.global_step)
                    plt.close('all')
                    gc.collect()
                if self.global_step % self.save_every_steps == 0:
                    self.save_checkpoint(epoch, self.global_step)

                self.global_step += 1

            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f'Epoch {epoch} - Loss: {epoch_loss:.4f}')

            if self.use_wandb:
                wandb.log({
                    'epoch_loss': epoch_loss,
                    'epoch': epoch,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })

            # Regular checkpoint every epoch (or every few epochs if desired)
            if epoch % 1 == 0:
                self.save_checkpoint(epoch, self.global_step)

        print("Training completed!")

if __name__ == "__main__":
    trainer = AudioVisualTrainer(
        video_dir='/home/cisco/nvmefudge/vggsound_1seconds',
        output_dir='./outputs',
        batch_size=48,
        num_epochs=100,
        learning_rate=2e-3,
        use_wandb=True,
        num_vis_samples=20,
        gradient_accumulation_steps=1,
        vis_every=5000,
        num_workers=15,
        force_new_training=False,
        unfreeze_hubert_epoch=0,
        unfreeze_vit_epoch=5,
        save_every_steps=3000
    )
    trainer.train()
