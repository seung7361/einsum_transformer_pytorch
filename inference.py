import torch
import torchvision
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_dataset
from model import GPT
from tqdm import tqdm
from PIL import Image

import time

### hyperparameters ###

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
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

batch_size = 32
learning_rate = 3e-4
num_epochs = 3

#######################

model = GPT(vocab_size, num_layers, embedding_dim, hidden_dim, num_heads, ff_dim, dropout).cuda()
print('model size: {:_} parameters'.format(sum(p.numel() for p in model.parameters())))

model.load_state_dict(torch.load('./checkpoints/epoch_0_step_10000.pt'))
model.eval()
print(model.generate('The', tokenizer))