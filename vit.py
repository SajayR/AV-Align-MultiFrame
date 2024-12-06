import torch
import torch.nn as nn
import timm

class ViTEmbedder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        
        # Load pretrained ViT
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Freeze the model parameters (optional, we can discuss if you wanna fine-tune)
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        x = self.model.forward_features(x)  # Get patch embeddings
        
        # Remove CLS token if present (first token)
        if hasattr(self.model, 'cls_token'):
            x = x[:, 1:, :]
            
        return x

# Test it out!
if __name__ == "__main__":
    vit = ViTEmbedder()
    # Print total number of parameters
    total_params = sum(p.numel() for p in vit.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print model architecture
    print("\nModel layers:")
    print(vit.model)
    
    # Create dummy batch of images
    batch = torch.randn(4, 3, 224, 224)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = vit(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {embeddings.shape}")
    # Should output something like: torch.Size([4, 196, 768])
    # 196 = 14x14 patches, 768 = embedding dimension