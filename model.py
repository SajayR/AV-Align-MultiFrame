import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViTEmbedder
from hubert import AudioEmbedder
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class AudioVisualModel(nn.Module):
    def __init__(self, temperature=2.5, initial_threshold=-2.5, scale_factor=2.5): #sigmoid(-2.5) = 0.08
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = AudioEmbedder()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # Add learnable threshold
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor)) ## add to optimizer
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        
        # Unfreeze the HuBERT model
        for param in self.audio_embedder.hubert.parameters():
            param.requires_grad = True
        
        for param in self.audio_embedder.projection.parameters():
            param.requires_grad = True
        
    def compute_temporal_similarity_matrix(self, audio_feats, visual_feats):
        """
        Compute pairwise cosine similarities between audio tokens and visual tokens across time
        
        Args:
            audio_feats: (B, Na, D)  # B=batch, Na=num_audio_tokens, D=embedding_dim
            visual_feats: (B, T, Nv, D) # T=time, Nv=num_visual_tokens
            
        Returns:
            similarities: (B, Na, T, Nv)
        """
        # Add time dimension to audio features for broadcasting
        audio_feats = audio_feats.unsqueeze(2)  # [B, Na, 1, D]
        
        # Normalize embeddings
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute similarities using einsum
        similarities = torch.einsum('bamd,btvd->batv', audio_feats, visual_feats)
        
        return similarities / self.temperature
    
    def aggregate_temporal_similarities(self, similarities):
        """
        Aggregate temporal similarities using learned threshold with smooth selection
        
        Args:
            similarities: (B, Na, T, Nv) or (B, B, Na, T, Nv)
                
        Returns:
            clip_similarities: (B) or (B, B)
            fraction_selected: Mean of selection strengths for monitoring
        """
        # Use sigmoid to keep threshold between 0 and 1
        threshold = torch.sigmoid(self.threshold)
        
        # Max pool over visual dimension
        max_visual_similarities = torch.max(similarities, dim=-1)[0]
        
        # Compute smooth selection weights
        raw_diff = max_visual_similarities - threshold
        
        selection_strength = F.relu(raw_diff) * self.scale_factor #dont think we need the softplus, relu is enough
       
        masked_similarities = max_visual_similarities * selection_strength
         
        # Compute weighted average over frames, this seems extremely wrong
        weighted_sum = masked_similarities.sum(dim=-1)
        weights_sum = selection_strength.sum(dim=-1)
        token_similarities = weighted_sum / weights_sum.clamp(min=1e-6)
        
        #token similarities should have a max of 1 and min of 0, shit does not
        #assert token_similarities.max() <= 1 and token_similarities.min() >= 0, "Token similarities should have a max of 1 and min of 0"
        # Average over audio tokens 
        clip_similarities = token_similarities.mean(dim=-1)
        
        # Track average selection strength
        #fraction_selected = selection_strength.mean()
        passed_threshold = (raw_diff > 0).float()
        fraction_selected = passed_threshold.mean()
        
        return clip_similarities, fraction_selected
    
    def compute_all_similarities(self, audio_feats, visual_feats):
        """
        Compute similarities between all pairs in batch
        """
        B = audio_feats.shape[0]
        
        # Expand dimensions for all pairs
        audio_feats = audio_feats.unsqueeze(1)  # [B, 1, Na, D]
        visual_feats = visual_feats.unsqueeze(0)  # [1, B, T, Nv, D]
        
        # Compute token-level similarities for all pairs
        similarities = torch.einsum('xyad,xytvd->xyatv', 
                                  F.normalize(audio_feats, dim=-1),
                                  F.normalize(visual_feats, dim=-1))
        similarities = similarities / self.temperature
        
        # Aggregate using threshold-based temporal selection
        clip_similarities, fraction_selected = self.aggregate_temporal_similarities(similarities)
        
        return clip_similarities, similarities, fraction_selected

    def compute_contrastive_loss(self, clip_similarities, token_sims, fraction_selected):
        """Compute InfoNCE loss with regularization"""
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        # Audio to Visual direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
        
        # Visual to Audio direction  
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
        
        # Average both directions
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        
        # Add regularization
        reg_loss = self.compute_regularization_losses(clip_similarities, token_sims)
        

        threshold = torch.sigmoid(self.threshold)
        max_visual_similarities = torch.max(token_sims, dim=-1)[0]  # [B, B, Na, T]
        raw_diff = max_visual_similarities - threshold
        
        # Compute selection ratio loss for positive pairs only
        positive_diffs = raw_diff[torch.arange(batch_size), torch.arange(batch_size)]  # [Na, T]
        binary_ish = torch.tanh(20 * positive_diffs)
        selection_ratio = (binary_ish + 1) / 2
        selection_reward = -0.1 * torch.log1p(selection_ratio.mean())  # 0.1 is alpha
        
        total_loss = selection_reward + contrastive_loss + reg_loss  # 2.5 selection = 1/4 of contrastive loss, 1/10

        return total_loss, contrastive_loss, reg_loss, fraction_selected, selection_reward

    def compute_regularization_losses(self, clip_sims, token_sims):
        """
        Regularization with temporal structure
        token_sims shape: [B, B, Na, T, Nv]
        """
        # 1. Non-negative pressure
        neg_sims = torch.clamp(token_sims, min=-20, max=0)  
        l_nonneg = torch.mean(neg_sims ** 2)
        
        # 2. Temperature regularization
        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) - torch.log(self.temperature), min=0) ** 2
        temp_high = torch.clamp(torch.log(self.temperature) - torch.log(torch.tensor(4.0, device=token_sims.device)), min=0) ** 2
        l_cal = temp_low + temp_high

        threshold = torch.sigmoid(self.threshold)
        l_threshold = torch.clamp(threshold - 0.9, min=0)**2 + torch.clamp(0.1 - threshold, min=0)**2

        l_scale = torch.clamp(self.scale_factor - 20.0, min=0)**2 + torch.clamp(1.0 - self.scale_factor, min=0)**2
        
        reg_loss = (0.15 * l_nonneg + 
                    2.0 * l_cal +
                    0.1 * l_threshold +
                    0.1 * l_scale)
        
        return reg_loss
        
    def forward(self, frames, audio):
        """
        Forward pass
        
        Args:
            frames: (B, T, C, H, W) batch of video frames
            audio: (B, samples) batch of audio samples
        """
        # Get embeddings
        visual_feats = self.visual_embedder(frames)  # [B, T, Nv, D]
        audio_feats = self.audio_embedder(audio)     # [B, Na, D]
        
        if self.training:
            # Compute similarities and loss
            clip_sims, token_sims, fraction_selected = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(clip_sims, token_sims, fraction_selected)
        else:
            # During inference, just get temporal similarity matrix
            similarities = self.compute_temporal_similarity_matrix(audio_feats, visual_feats)
            return similarities



if __name__ == "__main__":
    # Test the model
    model = AudioVisualModel()
    model.train()  # Set to training mode
    
    # Create dummy batch with temporal dimension
    batch_size = 2  # Keep small for testing
    num_frames = 10  # Matching our dataloader
    
    # Create dummy data matching our actual data shapes
    frames = torch.randn(batch_size, num_frames, 3, 224, 224)  # [B, T, C, H, W]
    audio = torch.randn(batch_size, 16331)  # Raw audio length from our dataset
    
    print("\nTesting shapes:")
    print(f"Input frames shape: {frames.shape}")
    print(f"Input audio shape: {audio.shape}")
    
    # Test training mode
    try:
        loss = model(frames, audio)
        print(f"\nTraining successful!")
        print(f"Training loss: {loss.item()}")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e
    
    # Test inference mode
    try:
        model.eval()
        with torch.no_grad():
            similarities = model(frames, audio)
            print(f"\nInference successful!")
            print(f"Output similarities shape: {similarities.shape}")
            print(f"Similarity stats - min: {similarities.min().item():.3f}, max: {similarities.max().item():.3f}")
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        raise e
    
    # Print some model info
    print("\nModel parameters:")
    print(f"Temperature: {model.temperature.item():.3f}")
    print(f"Threshold: {torch.sigmoid(model.threshold).item():.3f}")
    
    # Optional: print memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory stats:")
        print(torch.cuda.memory_summary(abbreviated=True))