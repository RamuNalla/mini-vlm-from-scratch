import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches, flattens them, and projects them 
    into a latent vector dimension (D).
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_dim: int = 64, image_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Number of patches per row/col and total number of patches (N)
        self.n_patches_per_side = image_size // patch_size
        self.num_patches = self.n_patches_per_side ** 2
        
        # Using a Conv2d layer is a clever and efficient way to implement patch extraction 
        # and linear projection in a single operation.
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (Batch_Size, Channels, Height, Width)
        Returns:
            tensor of shape (Batch_Size, Num_Patches, Embed_Dim)
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}x{W}) doesn't match model expected size ({self.image_size}x{self.image_size})"
        
        # (B, C, H, W) -> (B, Embed_Dim, H/patch_size, W/patch_size)
        x = self.projection(x)
        
        # (B, Embed_Dim, H', W') -> (B, Embed_Dim, N) where N = H' * W'
        x = x.flatten(2)
        
        # (B, Embed_Dim, N) -> (B, N, Embed_Dim)
        x = x.transpose(1, 2)
        return x
    

class ViTEmbeddings(nn.Module):
    """
    Combines Patch Embeddings, Learnable Classification Token ([CLS]), 
    and Learnable 1D Positional Embeddings.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_dim: int = 64, image_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, image_size)
        
        # Learnable Classification Token: prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

         # Learnable 1D Positional Embeddings (+1 accounts for the [CLS] token)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, emb_dim))
        
        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights properly (normal distribution)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (Batch_Size, Channels, Height, Width)
        Returns:
            tensor of shape (Batch_Size, Num_Patches + 1, Embed_Dim)
        """
        B = x.shape[0]
        
        # 1. Extract patch embeddings: (B, N, D)
        x = self.patch_embed(x)
        
        # 2. Expand [CLS] token to match the batch size: (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 3. Concatenate [CLS] token to the beginning of the patch sequence: (B, N+1, D)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. Add positional encodings (broadcasts across the batch dimension)
        x = x + self.pos_embed
        
        # 5. Apply dropout
        x = self.dropout(x)
        return x
    
