import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from model import GPT

### hyperparameters ###

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
vocab_size = tokenizer.vocab_size + 2 # eos, pad
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    'eos_token': '<|endoftext|>',
    'bos_token': '<|startoftext|>'
})

num_layers = 12
embedding_dim = 768
hidden_dim = 768
num_heads = 12
d_head = 64
ff_dim = 768 * 4
dropout = 0.1

assert d_head * num_heads == hidden_dim

#######################

model = GPT(vocab_size, num_layers, embedding_dim, hidden_dim, num_heads, ff_dim, dropout)
print('model size: {:_} parameters'.format(sum(p.numel() for p in model.parameters())))

print(model.generate('Hello, my name is', tokenizer))