import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
vocab_size = tokenizer.vocab_size + 2 # eos, pad
tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    'eos_token': '<|endoftext|>',
    'bos_token': '<|startoftext|>'
})
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
)

print(tokenizer('The', return_tensors='pt', padding='max_length', max_length=128, truncation=True)['input_ids'])