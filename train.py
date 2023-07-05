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

dataset = load_dataset('roneneldan/TinyStories')['train']['text'][:1000000]
print('started tokenizing...')
start = time.time()
train_dataset = []
for sentence in tqdm(dataset):
    train_dataset.append(tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=128, truncation=True)['input_ids'].squeeze(0))
print(f'finished tokenizing in {time.time() - start} seconds')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

def train(model, optimizer, scheduler, dataloader, batch_size, num_epochs):
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)
        pbar.set_description(f'Epoch: {epoch + 1}, Loss: -')

        step = 0
        for batch in pbar:
            optimizer.zero_grad()
            # batch: (batch_size, seq_len), dtype=torch.long
            # batch = tokenizer(batch, return_tensors='pt', padding='max_length', max_length=128, truncation=True)['input_ids'].cuda()
            batch = batch.cuda()

            input = batch[:, :-1].contiguous()
            label = batch[:, 1:].contiguous()

            output = model(input)

            loss = loss_fn(output.view(-1, vocab_size), label.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

            step += 1
            if step % 10000 == 0:
                torch.save(model.state_dict(), './checkpoints/epoch_{}_step_{}.pt'.format(epoch, step))
                print('saved checkpoint at epoch {} step {}'.format(epoch, step))
                print(model.generate('The', 100))

            if step % 100 == 0:
                transform = torchvision.transforms.ToPILImage()
                image = transform(model.positional_encoding.pos_encoding)
                image.save('./embedding/epoch_{}_step_{}.png'.format(epoch, step))
            

train(model, optimizer, scheduler, train_dataloader, batch_size, num_epochs)