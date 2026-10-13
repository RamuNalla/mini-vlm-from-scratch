import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module as described in the original Transformer paper.
    """
    def __init__(self, emb_dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
        # Ensure embedding dimension is divisible by the number of heads
        assert emb_dim % num_heads == 0, f"Embedding dimension ({emb_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = emb_dim // num_heads
        
        # Combined query, key, and value projections in a single linear layer for efficiency
        self.qkv_projection = nn.Linear(emb_dim, emb_dim * 3, bias=True)
        self.out_projection = nn.Linear(emb_dim, emb_dim, bias=True)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (Batch_Size, Num_Patches + 1, Embed_Dim)
        Returns:
            tensor of shape (Batch_Size, Num_Patches + 1, Embed_Dim)
        """
        B, N, D = x.shape
        
        # 1. Compute Q, K, V matrices simultaneously
        # Shape: (B, N, D * 3)
        qkv = self.qkv_projection(x)
        
        # 2. Split into Query, Key, and Value components: each of shape (B, N, D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 3. Reshape for multi-head attention: 
        # (B, N, num_heads, head_dim) -> transpose to (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Scaled Dot-Product Attention using PyTorch's optimized internal function
        # Computes: softmax((Q @ K.T) / sqrt(head_dim)) @ V
        # Shape: (B, num_heads, N, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        # 5. Re-assemble all heads together: 
        # (B, num_heads, N, head_dim) -> transpose to (B, N, num_heads, head_dim) -> flatten to (B, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # 6. Final linear output projection
        out = self.out_projection(out)
        return out