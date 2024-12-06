import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViTEmbedder
from dataset import ASTEmbedder

class AudioVisualModel(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = ASTEmbedder()
        self.temperature = temperature
        
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
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute similarities
        similarity = torch.bmm(audio_feats, visual_feats.transpose(1, 2))
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
        """
        Compute similarities between all pairs of audio and visual features in the batch
        
        Args:
            audio_feats: (B, Na, D)
            visual_feats: (B, Nv, D)
            
        Returns:
            all_similarities: (B, B) containing clip-level similarities for all pairs
        """
        B = audio_feats.shape[0]
        
        # Reshape to compute all pairs
        # (B, Na, D) -> (B, 1, Na, D) -> (B, B, Na, D)
        audio_feats = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        
        # (B, Nv, D) -> (1, B, Nv, D) -> (B, B, Nv, D)
        visual_feats = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Normalize
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute token-level similarities for all pairs
        # Result: (B, B, Na, Nv)
        token_sims = torch.matmul(
            audio_feats, 
            visual_feats.transpose(2, 3)
        ) / self.temperature
        
        # Aggregate using max-mean strategy
        # Max over visual dimension: (B, B, Na)
        max_sims = torch.max(token_sims, dim=3)[0]
        
        # Mean over audio dimension: (B, B)
        clip_sims = torch.mean(max_sims, dim=2)
        
        return clip_sims
        
    def compute_contrastive_loss(self, clip_similarities):
        """
        Compute InfoNCE loss in both A->V and V->A directions
        
        Args:
            clip_similarities: (B, B) matrix where:
                - diagonal elements are positive pairs (same video)
                - off-diagonal elements are negative pairs (different videos)
        """
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        # Audio to Visual direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)
        loss_a2v = -log_prob_a2v[torch.arange(batch_size), labels].mean()
        
        # Visual to Audio direction
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)
        loss_v2a = -log_prob_v2a[torch.arange(batch_size), labels].mean()
        
        return loss_a2v + loss_v2a
        
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
        visual_feats = self.visual_embedder(frames)        # (B, Nv, D)
        audio_feats = self.audio_embedder(spectrograms)    # (B, Na, D)
        
        if self.training:
            # Compute all pairwise similarities in batch efficiently
            all_similarities = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(all_similarities)
        else:
            # During inference, just compute similarities between corresponding pairs
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