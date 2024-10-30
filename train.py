import os
import numpy as np
import torch
import torch.nn as nn
import math
from model import Model_args, GPT
import time

# Model parameters
block_size = 128  # Window size (GPT-2 uses 1024)
batch_size = 32  # To be determined based on memory usage
n_layer = 12
n_head = 6
n_embed = 768
bias = False
dropout = 0.0
dataset_path = './data/tinystories'
init_from = 'scratch'  # 'scratch' or 'resume' - whether to train from scratch or resume
checkpoint_save_dir = './checkpoints'
eval_iters = 200
eval_interval = 2000  # Evaluate and save checkpoint every n steps
# Learning rate decay
learning_rate = 6e-4
warmup_iters = 2000
lr_decay_iters = 8000
min_lr = 6e-5
# Optimizer parameters
max_iters = 6000  # Number of iterations for training
weight_decay = 1e-1
betas = (0.9, 0.95)
grad_clip = 1.0  # Gradient clipping
# System
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# Check if CUDA supports bfloat16 data type

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# torch.amp.autocast for mixed precision

# Dataloader
data_dir = os.path.join(dataset_path)


def get_batch(split):
    # According to the author of NanoGPT, memmap should be used for each batch to avoid memory leaks
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    # torch.randint(a, b, (size,)) generates 'size' random numbers in the range (a, b)
    x = torch.stack(
        [torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])  # Extract x, y from data based on ix
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    # torch.stack(inputs, dim=0), dim is the new concatenation dimension

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    # pin_memory() locks the tensor in memory, non_blocking=True makes data transfer non-blocking
    return x, y


model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

iter_num = 0  # Will be overridden to 0 if resumed
best_val_loss = 1e9

assert init_from == 'scratch' or init_from == 'resume'
if init_from == 'scratch':
    print("Training model from scratch")
    model_args['vocab_size'] = 50304  # GPT-2 tokenizer vocab size
    # Directly using GPT-2's vocab, in prepare.py, tokenization is done using tiktoken.get_encoding('gpt2')
    gpt_args = Model_args(**model_args)
    model = GPT(gpt_args)  # Create the model

elif init_from == 'resume':  # Resume training
    print("Resuming model training")
    ckpt_path = os.path.join(checkpoint_save_dir, 'checkpoint.pt')  # Path to read the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']  # Read model parameters from checkpoint
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gpt_args = Model_args(**model_args)
    model = GPT(gpt_args)
    state_dict = checkpoint['model']  # Model weights
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']  # Iteration steps
    best_val_loss = checkpoint['best_val_loss']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# Optimization: Mixed precision training, mostly using float16, with some float32

model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # Clear the checkpoint after reading


# NanoGPT also uses torch.compile optimization, not implemented here

def estimate_loss():
    model.eval()  # Evaluation mode, no gradient computation
    out = {}
    for split in ['train', 'val']:
        # Calculate loss for both training and validation sets
        # In NanoGPT, many arguments are passed using dicts
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # print(f"Evaluating iteration {k}")
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)  # x, targets
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Switch back to train mode after evaluation
    return out


# NanoGPT uses cosine annealing for learning rate decay
def get_lr(now_iter):
    if now_iter < warmup_iters:  # (1) Warmup phase, linear increase
        return learning_rate * now_iter / warmup_iters
    elif now_iter > lr_decay_iters:  # (2) Exceeding decay phase, reach min_lr
        return min_lr
    else:  # (3) Between warmup and decay, cosine annealing for lr decay
        rate = (now_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        # Calculate proportion (0, 1)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * rate)) * (learning_rate - min_lr)


# Training code
X, Y = get_batch('train')
t_before = time.time()

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Set learning rate

    if iter_num > 0 and iter_num % eval_interval == 0:
        # Evaluation
        loss_dict = estimate_loss()
        print(f"Iteration {iter_num}, train_loss: {loss_dict['train']}, val_loss: {loss_dict['val']}")
        best_val_loss = min(loss_dict['val'], best_val_loss)
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss
        }
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)
        torch.save(checkpoint, os.path.join(checkpoint_save_dir, 'checkpoint.pt'))
        print(f"Checkpoint saved at {checkpoint_save_dir}/checkpoint.pt")

    with ctx:
        logits, loss = model(X, Y)
        print(f"Iteration: {iter_num}, Loss: {loss.item()}")
        scaler.scale(loss).backward()
        # Use scaler, scale loss (FP16), backward computes scaled gradients (FP16)

    if grad_clip > 0.0:
        scaler.unscale_(optimizer)  # Unscale gradients back to FP32
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Gradient clipping to prevent gradient explosion
    scaler.step(optimizer)  # Use scaler to execute optimizer.step()
    scaler.update()  # Update scaler factor
    """
    For using scaler, see this article: https://zhuanlan.zhihu.com/p/348554267

    Mixed precision is used, but FP32 to FP16 conversion may overflow, so scaling is required.

    GradScaler works by multiplying the loss by a scale factor before backpropagation,
    so that the gradients are also scaled by the same factor.
    To prevent affecting the learning rate, gradients are unscaled before update.
    The steps are as follows:
        Maintain a FP32 copy of the model parameters
        In each iteration:
            Copy and convert to FP16 model
            Forward pass (FP16 model parameters)
            Multiply loss by scale factor
            Backward pass (FP16 model parameters and gradients)
            Multiply gradients by 1/scale factor
            Use FP16 gradients to update FP32 model parameters
    """
    optimizer.zero_grad(set_to_none=True)  # Free up memory

    t_after = time.time()
    dt = t_after - t_before
    t_before = t_after

    iter_num += 1
    if iter_num > max_iters:
        break
