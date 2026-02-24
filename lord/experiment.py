import math
import argparse
import random
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

sns.set_theme(style="whitegrid")

class NewtonSchulzLowRankDecay:
    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if self.target_keywords and not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append(param)
        
    @torch.no_grad()
    def step(self):
        for W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            transposed = False
            if r > c:
                X = X.T
                transposed = True
              
            norm = X.norm() + 1e-8
            X = X / norm
            
            Y = X
            I = torch.eye(min(r, c), device=X.device)
            
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            W.sub_(self.decay_rate * Y.to(orig_dtype))

@dataclass
class ModelConfig:
    vocab_size: int = 114
    dim: int = 128
    depth: int = 2
    heads: int = 4
    mlp_dim: int = 512
    use_qk_norm: bool = True 

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.head_dim = config.dim // config.heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(config.dim)
            self.k_norm = RMSNorm(config.dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
            
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).reshape(B, T, C))

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, config.dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': RMSNorm(config.dim),
                'attn': Attention(config),
                'norm2': RMSNorm(config.dim),
                'mlp': nn.Sequential(
                    nn.Linear(config.dim, config.mlp_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(config.mlp_dim, config.dim, bias=False)
                )
            }) for _ in range(config.depth)
        ])
        self.norm_final = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :T, :]
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['mlp'](layer['norm2'](x))
        x = self.norm_final(x)
        return self.lm_head(x[:, -1, :])

class ModularAdditionDataset(Dataset):
    def __init__(self, p=113, split='train', train_frac=0.5, seed=42):
        data = [(i, j, p, (i + j) % p) for i in range(p) for j in range(p)]
        random.seed(seed)
        random.shuffle(data)
        split_idx = int(len(data) * train_frac)
        self.data = data[:split_idx] if split == 'train' else data[split_idx:]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        i, j, eq, res = self.data[idx]
        return torch.tensor([i, j, eq], dtype=torch.long), torch.tensor(res, dtype=torch.long)

def get_stable_rank(model):
    ranks = []
    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name:
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = sum(sigma^2) / max(sigma)^2 = ||W||_F^2 / ||W||_2^2
            ranks.append((S.norm()**2 / (S[0]**2 + 1e-9)).item())
    return sum(ranks) / len(ranks) if ranks else 0

def train_run(args, train_frac, decay_type, decay_val, device):
    p = 113
    config = ModelConfig(vocab_size=p+1, use_qk_norm=True)
    model = Transformer(config).to(device)
    
    if decay_type == 'L2':
        # Standard L2 on everything
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay_val)
        lrd_opt = None
    else:
        decay_params, nodecay_params = [], []
        target = ["q_proj", "k_proj"]
        for name, p_val in model.named_parameters():
            if any(t in name for t in target): nodecay_params.append(p_val)
            else: decay_params.append(p_val)
            
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.1}, # Keep mild L2 for stability
            {'params': nodecay_params, 'weight_decay': 0.0}
        ], lr=1e-3)
        lrd_opt = NewtonSchulzLowRankDecay(model.named_parameters(), decay_rate=decay_val, target_keywords=target)

    # Data
    train_ds = ModularAdditionDataset(p=p, split='train', train_frac=train_frac)
    val_ds = ModularAdditionDataset(p=p, split='val', train_frac=train_frac)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    
    # Loop
    max_val_acc = 0.0
    history = {'step': [], 'val_acc': [], 'rank': []}
    
    consecutive_high_acc = 0
    
    pbar = tqdm(range(args.steps), desc=f"Train({decay_type}={decay_val}, Frac={train_frac})", leave=False)
    iter_loader = iter(train_loader)
    
    for step in pbar:
        try: x, y = next(iter_loader)
        except StopIteration: iter_loader = iter(train_loader); x, y = next(iter_loader)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrd_opt: lrd_opt.step()
        
        if step % 200 == 0:
            model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    corr += (model(vx).argmax(-1) == vy).sum().item()
                    tot += vy.size(0)
            val_acc = corr / tot
            model.train()
            
            rank = get_stable_rank(model)
            max_val_acc = max(max_val_acc, val_acc)
            
            history['step'].append(step)
            history['val_acc'].append(val_acc)
            history['rank'].append(rank)
            pbar.set_postfix({'acc': f"{val_acc:.2f}", 'rank': f"{rank:.2f}"})
            
            if val_acc > 0.99:
                consecutive_high_acc += 1
                if consecutive_high_acc >= 2:
                    break
            else:
                consecutive_high_acc = 0
                
    return max_val_acc, history, model

def run_phase_diagram(args):
    print(">>> Starting Phase Diagram Experiment (This may take time)...")
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # Log-spaced decay rates
    decay_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    
    results = {'L2': np.zeros((len(fractions), len(decay_rates))),
               'LowRank': np.zeros((len(fractions), len(decay_rates)))}
    
    for i, frac in enumerate(fractions):
        for j, rate in enumerate(decay_rates):
            # Run Baseline (L2)
            acc_l2, _, _ = train_run(args, frac, 'L2', rate * 10, args.device) # L2 scale adjustment
            results['L2'][i, j] = acc_l2
            
            # Run Ours (LowRank)
            acc_lr, _, _ = train_run(args, frac, 'LowRank', rate, args.device)
            results['LowRank'][i, j] = acc_lr

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, name in zip(axes, ['L2', 'LowRank']):
        im = ax.imshow(results[name], origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"{name} Decay Phase Diagram")
        ax.set_xlabel("Decay Strength")
        ax.set_ylabel("Training Data Fraction")
        ax.set_xticks(range(len(decay_rates)))
        ax.set_xticklabels([f"{d:.0e}" for d in decay_rates], rotation=45)
        ax.set_yticks(range(len(fractions)))
        ax.set_yticklabels(fractions)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig("phase_diagram.png", dpi=150)
    print("Phase diagram saved to 'phase_diagram.png'.")

def run_mechanism_analysis(args):
    print(">>> Starting Mechanism Analysis (Single detailed run)...")
    
    # 1. Train both models
    print("Training Baseline (L2)...")
    _, hist_l2, model_l2 = train_run(args, 0.5, 'L2', 0.1, args.device)
    
    print("Training Ours (LowRank)...")
    _, hist_lr, model_lr = train_run(args, 0.5, 'LowRank', 0.005, args.device)
    
    # 2. Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)
    
    # A. Validation Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hist_l2['step'], hist_l2['val_acc'], label='L2', alpha=0.7)
    ax1.plot(hist_lr['step'], hist_lr['val_acc'], label='LowRank', linewidth=2)
    ax1.set_title("Grokking Speed")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Val Acc")
    ax1.legend()
    
    # B. Stable Rank Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hist_l2['step'], hist_l2['rank'], label='L2', linestyle='--')
    ax2.plot(hist_lr['step'], hist_lr['rank'], label='LowRank', linewidth=2)
    ax2.set_title("Effective Rank of QK")
    ax2.set_xlabel("Steps")
    ax2.legend()
    
    # C. SVD Spectrum (Last Layer, Head 0)
    ax3 = fig.add_subplot(gs[0, 2:])
    def get_svd(model):
        W = model.layers[0]['attn'].q_proj.weight.detach()
        # Roughly check Q*K^T spectrum
        # Ideally we check the interaction matrix, simplified here to weight spectrum
        S = torch.linalg.svdvals(W.float()).cpu().numpy()
        return S / S[0] # Normalize
    
    ax3.plot(get_svd(model_l2), label='L2 (Spectrum)', marker='.', alpha=0.5)
    ax3.plot(get_svd(model_lr), label='LowRank (Spectrum)', marker='o', linewidth=2)
    ax3.set_yscale('log')
    ax3.set_title("Singular Value Spectrum (Normalized)")
    ax3.set_xlabel("Singular Value Index")
    ax3.legend()
    
    # D. Attention Pattern Visualization
    # We visualize the Token-Token interaction map: E @ Q @ K^T @ E^T
    # This shows the "algorithm" the model learned.
    def plot_attn(model, ax, title):
        p = 113
        device = args.device
        model.eval()
        tokens = torch.arange(p, device=device)
        
        # Manually run first layer parts
        emb = model.embedding(tokens) + model.pos_embedding[:, 0, :]
        layer = model.layers[0]
        x = layer['norm1'](emb)
        
        attn_layer = layer['attn']
        Q = attn_layer.q_norm(attn_layer.q_proj(x))
        K = attn_layer.k_norm(attn_layer.k_proj(x))
        
        # Reshape for head 0
        head_dim = model.config.dim // model.config.heads
        Q = Q.view(p, model.config.heads, head_dim)[:, 0, :]
        K = K.view(p, model.config.heads, head_dim)[:, 0, :]
        
        # Attention Score
        Attn = (Q @ K.T) / (head_dim**0.5)
        Attn = Attn.softmax(dim=-1).cpu().detach().numpy()
        
        if sns:
            sns.heatmap(Attn, ax=ax, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
        else:
            ax.imshow(Attn, cmap="viridis")
        ax.set_title(title)
        
    ax4 = fig.add_subplot(gs[1, 0:2])
    plot_attn(model_l2, ax4, "L2 Attention Pattern (Noisy)")
    
    ax5 = fig.add_subplot(gs[1, 2:4])
    plot_attn(model_lr, ax5, "LowRank Attention Pattern (Structured)")
    
    plt.tight_layout()
    plt.savefig("mechanism_analysis.png", dpi=150)
    print("Analysis results saved to 'mechanism_analysis.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mechanism', choices=['phase_diagram', 'mechanism'],
                        help="Choose experiment mode")
    parser.add_argument('--steps', type=int, default=4000, help="Training steps per run")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Running in mode: {args.mode} on {args.device}")
    
    if args.mode == 'phase_diagram':
        # Lower steps for grid search efficiency
        args.steps = 2500 
        run_phase_diagram(args)
    else:
        run_mechanism_analysis(args)