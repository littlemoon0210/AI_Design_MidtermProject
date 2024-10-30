import os
import tiktoken
import torch
from model import GPT, Model_args
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Sample from a trained model.")
parser.add_argument('--out_dir', type=str, default='./out', help='Directory to save generated samples')
parser.add_argument('--start', type=str, default='Once upon a time', help='Initial text for generation')
args = parser.parse_args()

# Parameter configuration
checkpoint_save_dir = './checkpoints'
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Generation parameters
top_k = 200
tempreture = 0.5  # Increase for more randomness
start = args.start  # Use the command-line argument as the initial text
num_samples = 1  # Number of samples to generate
max_new_tokens = 128

# Load checkpoint
print(f"Load checkpoint from {checkpoint_save_dir}")
ckpt_path = os.path.join(checkpoint_save_dir, 'checkpoint.pt')  # Read checkpoint path
checkpoint = torch.load(ckpt_path, map_location=device)
args_model = checkpoint['model_args']
model = GPT(Model_args(**args_model))

# Load weights
state_dict = checkpoint['model']

# Remove unwanted prefix
unwanted_prefix = '_orig_mod'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
decode = lambda x: enc.decode(x)
encode = lambda x: enc.encode(x, allowed_special={"<|endoftext|>"})

# Set initial input
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Start generating text
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, top_k=top_k, tempreture=tempreture)
            generated_text = decode(y[0].tolist())
            print(generated_text)
            print("----------")
            
            # Save to the specified output directory
            output_file = os.path.join(args.out_dir, f"sample_{k}.txt")
            os.makedirs(args.out_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
                print(f"Sample saved to {output_file}")
