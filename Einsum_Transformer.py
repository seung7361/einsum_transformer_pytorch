import torch

class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.W = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = torch.nn.Parameter(torch.randn(output_dim))
    
    def forward(self, x):
        # x: (..., input_dim)
        
        return torch.einsum('...i,io,o->...o', x, self.W, self.b)

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    
    def forward(self, x):
        x = x.long()

        return self.embedding[x]

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv = LinearLayer(hidden_dim, hidden_dim * 3)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        assert len(x.shape) == 3 # 3D Tensor
        # x: (batch_size, seq_len, hidden_dim)
        # 3 * hidden_dim == 3 * d_head * num_heads
        # b: batch_size, t: seq_len, d: d_head, k: 3, h: num_heads
        q, k, v = torch.einsum('b t (d k h) -> k b h t d', self.qkv(x))
        # q, k, v: (b h t d)

        wei = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (self.hidden_dim ** 0.5)
        # wei: (b h i j) where i == t, j == t
        wei = self.dropout(self.softmax(wei))
        wei = torch.einsum('b h i j, b h j d -> b h i d', wei, v)
        wei = torch.einsum('b h t d -> b t (h d)', wei)
        # wei: (b t D) where D == h d

        return wei


class PositionwiseFeedForwardLayer(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim

        self.ff1 = LinearLayer(hidden_dim, ff_dim)
        self.ff2 = LinearLayer(ff_dim, hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)

        x = self.ff1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.ff2(x)

        # x: (batch_size, seq_len, hidden_dim)

        return x