import os
import requests
import tiktoken
import numpy as np

# openai tiktoken
# https://zhuanlan.zhihu.com/p/629776230

input_file_path = os.path.join(os.path.dirname(__file__), 'TinyStories.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
#the dataset is too large.
train_data = data[:int(n*0.1)]
val_data = data[int(n*0.99):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

#train has 47,623,065 tokens
#val has 4,779,866 token