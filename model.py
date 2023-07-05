import torch
import einops

class Dropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape) > self.p).cuda()
            mask = mask.float()
            return x * mask
        else:
            return x

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, torch.zeros_like(x))

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.randn(hidden_dim))
        self.beta = torch.nn.Parameter(torch.randn(hidden_dim))
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.W = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = torch.nn.Parameter(torch.randn(output_dim))
    
    def forward(self, x):
        # x: (..., input_dim)
        # out: (..., output_dim)
        
        return torch.einsum('...i, i o, o->...o', x, self.W, self.b)

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
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        assert len(x.shape) == 3 # 3D Tensor
        # x: (batch_size, seq_len, hidden_dim)
        # 3 * hidden_dim == 3 * d_head * num_heads
        # b: batch_size, t: seq_len, d: d_head, k: 3, h: num_heads
        q, k, v = tuple(einops.rearrange(self.qkv(x), 'b t (d k h) -> k b h t d', k=3, h=self.num_heads))
        # q, k, v: (b h t d)

        out = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (self.hidden_dim ** 0.5)
        # out: (b h i j) where i == t, j == t
        out = self.dropout(torch.nn.functional.softmax(out, dim=-1))
        out = torch.einsum('b h i j, b h j d -> b h i d', out, v)
        out = einops.rearrange(out, 'b h t d -> b t (h d)')
        # out: (b t D) where D == h d

        return out

class PositionwiseFeedForwardLayer(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim

        self.W1 = torch.nn.Parameter(torch.randn(hidden_dim, ff_dim))
        self.b1 = torch.nn.Parameter(torch.randn(ff_dim))

        self.W2 = torch.nn.Parameter(torch.randn(ff_dim, hidden_dim))
        self.b2 = torch.nn.Parameter(torch.randn(hidden_dim))

        self.relu = ReLU()
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        # x, out: (batch_size, seq_len, hidden_dim)
        x  = self.relu(torch.einsum('...i, i o, o -> ...o', x, self.W1, self.b1))
        x = self.dropout(torch.einsum('...i, i o, o -> ...o', x, self.W2, self.b2))

        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length=128):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.embedding = torch.nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.pos_encoding = torch.nn.Parameter(torch.randn(max_length, embedding_dim))

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = x.long()
        B, T = x.shape

        emb = self.embedding[x]
        pos_emb = self.pos_encoding[:T, :]
        # emb, pos_emb: (batch_size, seq_len, embedding_dim)

        return emb + pos_emb

class DecoderLayer(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.layer_norm1 = LayerNorm(hidden_dim)
        self.layer_norm2 = LayerNorm(hidden_dim)
        self.layer_norm3 = LayerNorm(hidden_dim)

        self.multi_head_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.positionwise_feed_forward = PositionwiseFeedForwardLayer(hidden_dim, ff_dim, dropout)

        self.dropout = Dropout(dropout)
        self.relu = ReLU()
    
    def forward(self, x):
        # x, out: (batch_size, seq_len, hidden_dim)
        x = x.cuda()
        out = self.layer_norm1(x + self.dropout(self.multi_head_attention(x))) # MBA, Add & Norm
        out = self.layer_norm2(out + self.dropout(self.positionwise_feed_forward(out))) # FFN, Add & Norm

        return out

class DecoderBlock(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_dim, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.layers = torch.nn.ModuleList([
            DecoderLayer(vocab_size, embedding_dim, hidden_dim, num_heads, ff_dim, dropout) for _ in range(self.num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x

class GPT(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_dim, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.positional_encoding = PositionalEncoding(vocab_size, embedding_dim)
        self.decoder_block = DecoderBlock(vocab_size, num_layers, embedding_dim, hidden_dim, num_heads, ff_dim, dropout)

        self.out1 = torch.nn.Parameter(torch.randn(hidden_dim, vocab_size))
        self.out2 = torch.nn.Parameter(torch.randn(vocab_size))
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.positional_encoding(x)
        # x: (batch_size, seq_len, embedding_dim)

        x = self.decoder_block(x)
        # x: (batch_size, seq_len, hidden_dim)

        x = torch.einsum('...i, i j, j -> ...j', x, self.out1, self.out2)

        return x

    def generate(self, text, tokenizer, max_length=64, top_p=0.9):
        input_ids = tokenizer(text, return_tensors='pt')['input_ids']

        for i in range(max_length):
            outputs = self(input_ids)
            next_token_logits = outputs[0][-1, :]
            
            # top-p sampling
            # apply a softmax to convert the logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # sort the probabilities in descending order and compute their cumulative sum
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # sample from the filtered distribution
            next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # stop when end-of-text token is generated
            if next_token_id == tokenizer.pad_token_id or next_token_id == tokenizer.eos_token_id:
                break
        
        return tokenizer.decode(input_ids[0])