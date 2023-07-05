import torch
from model import GPT

num_layers = 12
embedding_dim = 768
hidden_dim = 768
num_heads = 12
d_head = 64
ff_dim = 768 * 4
dropout = 0.1

model1 = GPT(vocab_size=50257, num_layers=num_layers, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout).decoder_block
model2 = torch.nn.TransformerDecoder(
    decoder_layer=torch.nn.TransformerDecoderLayer(
        d_model=embedding_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation='relu'
    ),
    num_layers=num_layers
)

print("{:_}".format(sum(p.numel() for p in model1.parameters() if p.requires_grad)))
print("{:_}".format(sum(p.numel() for p in model2.parameters() if p.requires_grad)))

model1.cuda()
model2.cuda()

x1 = torch.randn(1, 1024, 768).cuda()
x2 = torch.randn(1024, 1, 768).cuda()

y1 = model1(x1)
y2 = model2(x2, x1)

print(y1.shape)
print(y2.shape)