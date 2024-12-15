import torch
import torch.nn as nn
import timm

class ViTEmbedder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') #torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        self.projection = nn.Linear(384, 512)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        # Get intermediate features
        x = self.model.get_intermediate_layers(x, n=1)[0]
        x = self.projection(x)
        
        # x now includes both CLS token and patch tokens
        # Remove CLS token (first token) if you want just patches
        #patch_tokens = x[:, 1:, :]  # Remove CLS token
        
        return x

# Test it out!
if __name__ == "__main__":
    vit = ViTEmbedder().to("cuda")
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in vit.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print model architecture
    print("\nModel layers:")
    print(vit.model)
    
    # Create dummy batch of images
    batch = torch.randn(4, 3, 224, 224, device="cuda")
    
    # Get embeddings
    with torch.no_grad():
        embeddings = vit(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {embeddings.shape}")
    # Should output something like: torch.Size([4, 196, 768])
    # 196 = 14x14 patches, 768 = embedding dimension