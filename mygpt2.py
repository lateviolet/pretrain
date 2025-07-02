import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
from tqdm import tqdm
import tiktoken
import os


@dataclass
class GPTConfig:
    
    block_size: int = 1024 # max sentence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    
class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class CasualSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # q, k, v for all heads
        # in a batch to accelerate
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.c_proj.FLAG_SCALING_SQRT = 1 # flagging that we need to scale it when initializing
        
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size).view(
                1, 1, config.block_size, config.block_size))) # attention mask (upper triangle)
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, channel nums = n_heads * head_size
        qkv = self.c_attn(x) # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # (B, T, nh, ns) -> (B, nh, T, ns)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        """
        # (B, nh, T, ns) @ (B, nh, ns, T) = (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, ns) = (B, nh, T, ns)
        """
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Using flash attention instead
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y) 
        return y
    
    
class Block(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class MyDataLoader:
    
    def __init__(self, B, T, file_path, max_batch=50, 
                 process_rank=0, world_size=1, mode='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size
        self.max_batch = max_batch
        
        with open(file_path, 'r') as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.max_len = len(self.tokens)
        
        print(f"{len(tokens)} tokens loaded, 1 epoch = {len(tokens) // (B*T)} batches")
        
        self.curr_pos = process_rank * B * T
        self.curr_batch = 0
        
    def __len__(self):
        return self.max_batch
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.curr_batch += 1
        if self.curr_batch > self.max_batch:
            raise StopIteration
        
        B, T = self.B, self.T
        assert self.max_len > B * T * self.world_size
        
        if self.curr_batch >= 2:
            self.curr_pos += B * T * self.world_size
        self.curr_pos %= self.max_len
        buf = self.tokens[self.curr_pos : self.curr_pos + B*T + 1]
        if self.curr_pos + B*T + 1 >= self.max_len:
            margin = (B*T + 1) - len(buf)
            buf = torch.cat((buf, self.tokens[:margin]), dim=0)
        assert len(buf) == B*T + 1
        
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        return x, y        
        

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # copy the reference, keeping them the same all the time
        self.transformer['wte'].weight = self.lm_head.weight
        self.apply(self.my_init_weights)
        
    def my_init_weights(self, module: nn.Module, std=0.02): # align with the gpt2
        if hasattr(module, 'FLAG_SCALING_SQRT'):
            std = (2 * self.config.n_layer) ** -0.5
            print(f"initializing {sum(p.numel() for p in module.parameters())} params",
                  f"in {module._get_name()} with sqrt scaling")
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            
    def configure_optimizer(self, weight_decay, lr, device):
        param_dict = {p_name: p for p_name, p in self.named_parameters()}
        param_dict = {p_name: p for p_name, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_info = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        import inspect
        fused_ok = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_ok and 'cuda' in device
        print(f"using fused AdamW: {used_fused}")
        optimizer = torch.optim.AdamW(optim_info, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
        return optimizer
        
    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pre-trained %s" % model_name)
        
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_name]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard mask
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        
        w_to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 
                          'mlp.c_fc.weight', 'mlp.c_proj.weight'] # weights should be transposed
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        
        for k in sd_keys_hf: # copy weights from hf model
            if any(k.endswith(w) for w in w_to_transpose):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def forward(self, idx, targets=None): # (B, T)
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"Connot forward sequence length {T} \
            longer than block size {self.config.block_size}"
            
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer['wpe'](pos)
        tok_embd = self.transformer['wte'](idx)
        x = tok_embd + pos_embd
        
        for block in self.transformer['h']:
            x = block(x)
            
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
        
    
def main(): 
    """
    model = GPT.from_pretrained('gpt2')
    model.to(device)
    print("successfully loaded pre-trained gpt2")
    """
    
    # Using DDP
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as torchDDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        print(f"Using DDP, this is process {ddp_local_rank}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    torch.set_float32_matmul_precision('high') 
    model = GPT(GPTConfig(vocab_size=50304)) # fix a ugly number to fit into GPU
    model.to(device)
    print(f"training on {device}")
    model = torch.compile(model, backend='eager')
    if ddp:
        model = torchDDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    
    max_step = 19073
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    
    # optimized lr selection
    def lr_curve(curr_step):
        if curr_step < warmup_steps: # linear warmup
            return max_lr * (curr_step + 1) / warmup_steps
        if curr_step > max_step: # using min_lr
            return min_lr
        
        # using cosine lr scheduler
        ratio = (curr_step - warmup_steps) / (max_step - warmup_steps)
        assert 0 <= ratio and ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio)) # range 0 to 1
        return min_lr + coeff * (max_lr - min_lr)
    
    B = 4
    T = 384 # adjust to fit in GPU memory
    gpt_batch = 524288
    ministep_count = gpt_batch // (B * T)
    
    dataloader = MyDataLoader(B=B, T=T, file_path='pretrain/input.txt',
                              max_batch=max_step * ministep_count // ddp_world_size, 
                              process_rank=ddp_local_rank, world_size=ddp_world_size)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # GPT-2 params
    optimizer = raw_model.configure_optimizer(weight_decay=0.1, lr=max_lr, device=device)
    
    step = 0
    ministep = 0
    loss_accum = 0.0
    for x, y in tqdm(dataloader, desc="training process"):
        ministep += 1
        x = x.to(device); y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / ministep_count
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (ministep == ministep_count - 1)
        loss.backward()
        if ministep == ministep_count:
            if ddp:
                torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
            step += 1
            ministep = 0
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_curve(step)
            print(f"\nstep {step}: loss = {loss_accum.item()}")
            loss_accum = 0.0
        
    if ddp:
        destroy_process_group()
            
if __name__ == '__main__':
    main()