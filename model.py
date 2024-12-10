import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViTEmbedder
from dataset import ASTEmbedder

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioVisualModel(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = ASTEmbedder()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def compute_similarity_matrix(self, audio_feats, visual_feats):
        """
        Compute pairwise cosine similarities between audio and visual tokens
        
        Args:
            audio_feats: (B, Na, D)  # B=batch, Na=num_audio_tokens, D=embedding_dim
            visual_feats: (B, Nv, D) # Nv=num_visual_tokens
            
        Returns:
            similarity_matrix: (B, Na, Nv)
        """
        # Normalize embeddings
        #print("Audio feats stats before norm - min:", audio_feats.min().item(), "max:", audio_feats.max().item())
        #print("Visual feats stats before norm - min:", visual_feats.min().item(), "max:", visual_feats.max().item())
        
        # Normalize embeddings
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute similarities and check values
        similarity = torch.bmm(audio_feats, visual_feats.transpose(1, 2))
        #print("Raw similarity stats - min:", similarity.min().item(),
          #  "max:", similarity.max().item())
        
        return similarity / self.temperature
    
    def aggregate_token_similarities(self, similarity_matrix):
        """
        Aggregate token-level similarities using max-mean strategy
        
        Args:
            similarity_matrix: (B, Na, Nv)
            
        Returns:
            clip_similarity: (B)
        """
        # Max pool over visual dimension for each audio token
        max_similarities = torch.max(similarity_matrix, dim=2)[0]  # (B, Na)
        
        # Average over audio tokens
        clip_similarity = torch.mean(max_similarities, dim=1)  # (B)
        return clip_similarity
    
    def compute_all_similarities(self, audio_feats, visual_feats):
        """Compute similarities between all pairs of audio and visual features in batch"""
        B = audio_feats.shape[0]
        
        audio_feats = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        visual_feats = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Normalize features
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute token-level similarities
        token_sims = torch.matmul(
            audio_feats, 
            visual_feats.transpose(2, 3)
        ) / self.temperature
        
        # Aggregate using max-mean strategy
        max_sims = torch.max(token_sims, dim=3)[0]  # Max over visual dimension
        clip_sims = torch.mean(max_sims, dim=2)     # Mean over audio dimension
        
        return clip_sims, token_sims

    def compute_contrastive_loss(self, clip_similarities, token_sims):
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
        
        total_loss = contrastive_loss + reg_loss
        
        return total_loss
    
    def compute_regularization_losses(self, clip_sims, token_sims):
        """Compute regularization terms"""
        
        # 1. Non-negative pressure - encourage positive evidence
        neg_sims = torch.clamp(token_sims, max=0)  # Only keep negative similarities
        l_nonneg = torch.mean(neg_sims ** 2)
        
        # 2. Temperature/Calibration stability
        l_cal = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) - torch.log(self.temperature), min=0) ** 2
        
        # 3. Spatial smoothness on attention maps (simplified TV loss)
        spatial_diffs = token_sims[..., 1:] - token_sims[..., :-1]
        l_spatial = torch.mean(spatial_diffs ** 2)
        
        # 4. Sparsity loss - encourage focused attention
        # Normalize attention scores to [0,1] range for each audio token
        attn_norm = torch.sigmoid(token_sims)  # [B, B, Na, Nv]
        
        # Parameters for sparsity control
        threshold = 0.5  # Attention threshold
        alpha = 2.0      # Exponential penalty strength
        
        # Count patches above threshold with soft thresholding
        above_threshold = F.relu(attn_norm - threshold)
        # Sum across visual patches for each audio token
        num_high_attn = torch.sum(above_threshold, dim=-1)  # [B, B, Na]
        # Apply exponential penalty
        l_sparsity = torch.mean(torch.exp(alpha * num_high_attn))
        
        # Combine regularization terms
        reg_loss = (0.01 * l_nonneg + 
                    0.1 * l_cal + 
                    0.01 * l_spatial + 
                    0.01 * l_sparsity)  # Can adjust this weight to control sparsity strength
                    
        return reg_loss
        
    def forward(self, frames, spectrograms):
        """
        Forward pass computing embeddings, similarities and loss
        
        Args:
            frames: (B, C, H, W) batch of video frames
            spectrograms: (B, T, F) batch of audio spectrograms
            
        Returns:
            loss if training, clip_similarities if not
        """
        # Get embeddings
        visual_feats = self.visual_embedder(frames)
        audio_feats = self.audio_embedder(spectrograms)
        
        if self.training:
            # Get similarities and token-level similarities
            clip_sims, token_sims = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(clip_sims, token_sims)
        else:
            # During inference, just get clip similarities
            token_sims = self.compute_similarity_matrix(audio_feats, visual_feats)
            return self.aggregate_token_similarities(token_sims)

if __name__ == "__main__":
    # Test the model
    model = AudioVisualModel()
    
    # Create dummy batch
    batch_size = 4
    frames = torch.randn(batch_size, 3, 224, 224)
    specs = torch.randn(batch_size, 300, 128)
    
    # Test training mode
    loss = model(frames, specs)
    print(f"Training loss: {loss.item()}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        similarities = model(frames, specs)
        print(f"Inference similarities shape: {similarities.shape}")  # Should be (batch_size)
        print(f"Similarity values: {similarities}")