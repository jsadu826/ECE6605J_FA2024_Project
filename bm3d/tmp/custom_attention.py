import torch
import torch.nn as nn

class CustomSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, similarity_matrix, scaling_factor=1.0):
        super(CustomSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.similarity_matrix = similarity_matrix  # (num_patches, num_similar, 2)
        self.scaling_factor = scaling_factor

        # Learnable projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # Batch size, Number of patches, Embedding dimension
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        # Add bias from similarity matrix
        for i in range(N):  # Iterate over each patch
            similar_patches = self.similarity_matrix[i]  # (5, 2)
            for (x_idx, y_idx) in similar_patches:
                if x_idx < N and y_idx < N:  # Ensure indices are valid
                    attn_scores[:, :, i, x_idx] += self.scaling_factor
                    attn_scores[:, :, i, y_idx] += self.scaling_factor

        # Apply softmax to obtain attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        # Project back to embedding space
        return self.out_proj(attn_output)

# Example usage
similarity_matrix = torch.randint(0, 8, (64, 5, 2))  # Randomly generated similarity matrix
x = torch.rand(1, 64, 768)  # Input tensor: Batch size 1, 64 patches, 768 embedding dim

custom_attention = CustomSelfAttention(embed_dim=768, num_heads=8, similarity_matrix=similarity_matrix, scaling_factor=0.1)
output = custom_attention(x)
