"""
train.py — Minimal MLP training harness for normalization experiments.

Keeps the experiment loop clean so notebooks can focus on the norm layer itself.
Supports swappable normalization via a factory function passed at construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List, Tuple, Dict
import numpy as np


# ─────────────────────────────────────────────
# MLP WITH PLUGGABLE NORM
# ─────────────────────────────────────────────

class NormMLP(nn.Module):
    """
    A simple L-layer MLP where the normalization layer is swappable.
    Architecture: Linear → Norm → Activation → ... → Linear → output

    norm_factory: callable(num_features) → norm module (or None for no norm)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 4,
        norm_factory: Optional[Callable] = None,
        activation: str = 'tanh',
    ):
        super().__init__()
        self.layers    = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.norm_factory = norm_factory
        self.activation   = activation

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=(norm_factory is None)))
            if i < len(dims) - 2 and norm_factory is not None:
                self.norms.append(norm_factory(dims[i + 1]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (logits, list_of_pre_activation_tensors_per_layer)"""
        activations = []

        for i, linear in enumerate(self.layers[:-1]):
            x = linear(x)
            if self.norms:
                x = self.norms[i](x)
            activations.append(x.detach().clone())
            x = torch.tanh(x) if self.activation == 'tanh' else F.relu(x)

        x = self.layers[-1](x)
        return x, activations


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    lr: float = 0.01,
    n_steps: int = 1000,
    batch_size: int = 32,
    log_every: int = 100,
    collect_grad_every: int = 10,
) -> Dict:
    """
    Simple SGD training loop. Returns a history dict with:
      - losses
      - activation snapshots (list of lists, one per log_every step)
      - per-layer gradient magnitudes at each collect_grad_every step
      - mean/std of pre-norm activations (for covariate shift plots)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    N = X.shape[0]

    history = {
        'losses':        [],
        'activations':   [],   # list of [layer_acts, ...] snapshots
        'grad_norms':    [],   # list of [grad_per_layer]
        'act_means':     [],   # (step, num_features) for covariate shift
        'act_stds':      [],
    }

    for step in range(n_steps):
        # mini-batch
        idx  = torch.randint(0, N, (batch_size,))
        xb   = X[idx]
        yb   = Y[idx]

        logits, acts = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['losses'].append(loss.item())

        if step % log_every == 0:
            history['activations'].append([a.clone() for a in acts])
            print(f"step {step:5d} | loss {loss.item():.4f}")

        if step % collect_grad_every == 0:
            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.abs().mean().item())
            history['grad_norms'].append(grad_norms)

            if acts:
                history['act_means'].append(acts[0].mean(dim=0).numpy())
                history['act_stds'].append(acts[0].std(dim=0).numpy())

    history['act_means'] = np.array(history['act_means'])
    history['act_stds']  = np.array(history['act_stds'])
    return history


# ─────────────────────────────────────────────
# SYNTHETIC DATASET
# ─────────────────────────────────────────────

def make_dataset(
    n_samples: int = 2000,
    n_features: int = 16,
    n_classes: int = 4,
    covariate_shift: bool = False,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic classification dataset.

    If covariate_shift=True, features have very different scales
    (simulating what happens before normalization kicks in).
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)

    if covariate_shift:
        # exaggerate scale differences across features
        scales = torch.logspace(-1, 2, n_features)  # 0.1 to 100
        X = X * scales

    # random linear decision boundary
    W = torch.randn(n_features, n_classes)
    logits = X @ W
    Y = logits.argmax(dim=1)
    return X, Y


# ─────────────────────────────────────────────
# COMPARE MULTIPLE NORMS
# ─────────────────────────────────────────────

def compare_norms(
    norm_factories: Dict[str, Optional[Callable]],
    X: torch.Tensor,
    Y: torch.Tensor,
    hidden_dim: int = 64,
    n_layers: int = 4,
    lr: float = 0.01,
    n_steps: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Train an MLP with each norm variant and return all histories.
    Use for comparing loss curves, gradient flow, covariate shift side-by-side.
    """
    in_dim  = X.shape[1]
    n_classes = Y.max().item() + 1
    results = {}

    for name, factory in norm_factories.items():
        print(f"\n{'='*40}")
        print(f"  Training with: {name}")
        print(f"{'='*40}")
        torch.manual_seed(seed)
        model = NormMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=n_classes,
            n_layers=n_layers,
            norm_factory=factory,
        )
        hist = train(model, X, Y, lr=lr, n_steps=n_steps, batch_size=batch_size)
        results[name] = hist

    return results
