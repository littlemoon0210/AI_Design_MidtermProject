import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import inspect

# Model parameters
from dataclasses import dataclass


@dataclass
class Model_args:
    block_size: int = 1024  # Maximum input size
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0  # No dropout by default
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: slightly better and faster


class RMS_Norm(nn.Module):
    # RMS Norm, similar to that used in LLaMA
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps  # Avoid division by zero

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        sqrt_pow_mean = torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True))
        # Calculate L2 norm divided by n, then take square root as defined in RMS Norm
        return self.weight * hidden_states / (sqrt_pow_mean + self.eps)


class flash_att(nn.Module):
    # Flash attention, inspired by NanoGPT
    def __init__(self, args):
        super().__init__()
        # Combine qkv into a single Linear layer
        self.qkv_atten = nn.Linear(args.n_embed, 3 * args.n_embed, bias=args.bias)
        # According to one paper, head_size should equal seq_length for it to make sense
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        # Calculate head_size
        assert args.n_embed % args.n_head == 0
        self.head_size = args.n_embed // args.n_head
        # Dropout
        self.dropout = args.dropout  # This is the dropout probability, set to 0 in generate mode
        self.att_dropout = nn.Dropout(self.dropout)
        # Equivalent to nn.Dropout(p=self.dropout)
        # Projection layer
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=args.bias)

    def forward(self, x):
        B, T, C = x.shape
        # x shape: (B, T, C)
        q, k, v = self.qkv_atten(x).split(self.n_embed, dim=2)  # B, T, C

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Use PyTorch's built-in flash attention
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        # During training, dropout is applied
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Transpose to (B, T, nh, hs)
        y = y.transpose(1, 2)  # (B, T, nh, hs)
        # .contiguous() ensures the tensor is contiguous in memory
        y = y.contiguous().view(B, T, C)  # (B, T, C)

        # Output goes through projection layer and dropout
        return self.att_dropout(self.c_proj(y))


class MLP(nn.Module):
    # MLP inspired by the LLaMA MLP structure
    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.up_proj = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)
        self.down_c_proj = nn.Linear(4 * args.n_embed, args.n_embed, bias=args.bias)
        # Using ReLU
        self.act_func = nn.functional.relu
        # Adding a gating mechanism similar to LLaMA
        self.gate = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)

    def forward(self, x):
        # Unlike LLaMA code that slices input X, we don’t slice here
        gate_proj = self.gate(x)
        x = self.up_proj(x)

        # LLaMA code:
        # intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        # NanoGPT code:
        # x = self.act_func(x)
        # Main difference: NanoGPT applies activation to up_proj(x), while LLaMA applies it to gate_proj

        x = self.act_func(gate_proj) * x  # Element-wise multiplication with the gate
        x = self.down_c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    # The block to be stacked later
    def __init__(self, args):
        super().__init__()
        self.norm = RMS_Norm(args.n_embed)
        self.attn = flash_att(args)
        self.mlp = MLP(args)

    def forward(self, x):
        # Pre-normalization
        x = x + self.attn(self.norm(x))  # Residual connection
        return x + self.mlp(self.norm(x))  # Residual connection


class GPT(nn.Module):
    # A hybrid of LLaMA and GPT-2
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embed),
            # Token embedding
            wpe=nn.Embedding(args.block_size, args.n_embed),
            # Learnable positional embedding
            drop=nn.Dropout(args.dropout),
            h=nn.ModuleList([Block(args) for i in range(args.n_layer)]),
            norm=RMS_Norm(args.n_embed)
        ))

        self.lm_head = nn.Linear(args.n_embed, args.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # This is not a simple assignment, wte and lm_head share parameters
        # lm_head (n_embed, vocab_size) predicts tokens from embeddings
        # wte ()

        self.apply(self._init_weights)  # Initialize weights
        n_sum = 0
        # Normal distribution initialization for attention projection layers and MLP down-sampling
        for pname, p in self.named_parameters():
            n_sum += p.numel()  # Count parameters
            if pname.endswith('c_proj.weight'):  # c_proj is the context-aware projection layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layer))

        print(f"Number of model parameters: {n_sum}")

    def _init_weights(self, module):  # Initialize first layer and embedding
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):  # Targets are used for cross-entropy loss during training
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # Position

        # Embedding
        token_embed = self.transformer.wte(idx)  # (B, T, n_embed)
        pos_embed = self.transformer.wpe(pos)  # (t, n_embed)
        # Positional embedding is learnable

        x = self.transformer.drop(token_embed + pos_embed)  # Combine token and position embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)

        # Through lm_head
        # targets=True indicates training phase and loss computation
        # logits take the last (-1) to align with target token dimension for loss calculation

        if targets is not None:
            logits = self.lm_head(x)
            # Use -1 to take the last dimension, drop preceding dimensions
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)  # Cross-entropy loss
        else:  # For generation
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Create a dict mapping parameter names to parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Remove parameters that do not require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Weight decay
        # Apply weight decay to 2D parameters, not others, splitting into two groups
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Print number of decay and non-decay parameters
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Number of parameters with weight decay: {num_decay}, without weight decay: {num_nodecay}")

        # Create an AdamW optimizer, check for fused support
        # Check if AdamW's parameters include 'fused' for optimization
        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device_type == 'cuda'  # Also requires GPU
        if use_fused:
            print("AdamW optimizer uses fused version!")
        extra_args = {'fused': True} if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # betas: coefficients for computing running averages of gradient and its square

        return optimizer

    def generate(self, idx, max_generate_tokens, tempreture=1.0, top_k=None):
        # Top-p, top-k, and temperature concepts
        # max_generate_tokens is the maximum number of new tokens to generate
        for _ in range(max_generate_tokens):
            idx = idx if idx.shape[1] <= self.args.block_size else idx[:, -self.args.block_size:]
            # Truncate to block_size if exceeds maximum size
            # Not sure how idx is padded to block_size when length is less than block_size
            logits, _ = self(idx)
            logits = logits[:, -1, :] / tempreture  # (B, T, C), take the last one as the generated token
            # Higher temperature increases randomness
            # Softmax properties: smaller values reduce differences in token probs, increasing randomness

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # Ignore tokens beyond top_k

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample based on probs
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
