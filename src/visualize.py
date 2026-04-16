"""
visualize.py — Reusable plotting utilities for normalization experiments.

All functions return matplotlib Figure objects so notebooks can call
fig.savefig() or just display them inline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional


PALETTE = {
    'batchnorm': '#4C72B0',
    'layernorm': '#DD8452',
    'rmsnorm':   '#55A868',
    'nonorm':    '#C44E52',
    'ghost':     '#8172B2',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F8F8',
    'axes.grid':        True,
    'grid.color':       'white',
    'grid.linewidth':   1.2,
    'font.family':      'monospace',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})


# ─────────────────────────────────────────────
# ACTIVATION DISTRIBUTION
# ─────────────────────────────────────────────

def plot_activation_distributions(
    activations_dict: Dict[str, List[torch.Tensor]],
    layer_idx: int = -1,
    title: str = "Activation Distributions Across Layers"
) -> plt.Figure:
    """
    Plot histograms of activations for multiple norm strategies at a given layer.

    activations_dict: {'batchnorm': [layer0_acts, layer1_acts, ...], ...}
    layer_idx: which layer's activations to plot (-1 = last)
    """
    norms = list(activations_dict.keys())
    fig, axes = plt.subplots(1, len(norms), figsize=(5 * len(norms), 4), sharey=True)
    if len(norms) == 1:
        axes = [axes]

    for ax, norm_name in zip(axes, norms):
        acts = activations_dict[norm_name][layer_idx].detach().flatten().numpy()
        color = PALETTE.get(norm_name, '#888888')
        ax.hist(acts, bins=60, color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.set_title(norm_name, fontweight='bold', color=color)
        ax.set_xlabel("Activation value")
        mean, std = acts.mean(), acts.std()
        ax.axvline(mean, color='black', linestyle='--', linewidth=1.2, label=f'μ={mean:.2f}')
        ax.axvline(mean + std, color='gray', linestyle=':', linewidth=1, label=f'σ={std:.2f}')
        ax.axvline(mean - std, color='gray', linestyle=':', linewidth=1)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GRADIENT FLOW
# ─────────────────────────────────────────────

def plot_gradient_flow(
    grad_histories: Dict[str, List[float]],
    title: str = "Gradient Magnitude Across Layers"
) -> plt.Figure:
    """
    Line plot of mean gradient magnitude per layer.

    grad_histories: {'batchnorm': [grad_l0, grad_l1, ...], 'nonorm': [...], ...}
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for norm_name, grads in grad_histories.items():
        color = PALETTE.get(norm_name, '#888888')
        ax.plot(grads, marker='o', markersize=4, linewidth=2,
                color=color, label=norm_name)

    ax.set_xlabel("Layer index (0 = closest to output)")
    ax.set_ylabel("Mean |gradient|")
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# TRAINING LOSS CURVES
# ─────────────────────────────────────────────

def plot_training_curves(
    loss_histories: Dict[str, List[float]],
    title: str = "Training Loss Comparison"
) -> plt.Figure:
    """
    Overlaid training loss curves for multiple normalization strategies.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for norm_name, losses in loss_histories.items():
        color = PALETTE.get(norm_name, '#888888')
        ax.plot(losses, linewidth=2, color=color, label=norm_name, alpha=0.9)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# COVARIATE SHIFT
# ─────────────────────────────────────────────

def plot_covariate_shift(
    mean_histories: Dict[str, np.ndarray],
    std_histories: Dict[str, np.ndarray],
    feature_idx: int = 0,
    title: str = "Internal Covariate Shift (feature 0)"
) -> plt.Figure:
    """
    Plot how the mean and std of a single feature evolve over training steps.

    mean_histories: {'batchnorm': array of shape (steps,), ...}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for norm_name in mean_histories:
        color = PALETTE.get(norm_name, '#888888')
        ax1.plot(mean_histories[norm_name][:, feature_idx],
                 color=color, linewidth=1.8, label=norm_name)
        ax2.plot(std_histories[norm_name][:, feature_idx],
                 color=color, linewidth=1.8, label=norm_name)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title("Feature mean over training", fontweight='bold')
    ax1.set_xlabel("Step"); ax1.set_ylabel("Mean")
    ax1.legend()

    ax2.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title("Feature std over training", fontweight='bold')
    ax2.set_xlabel("Step"); ax2.set_ylabel("Std")
    ax2.legend()

    fig.suptitle(title, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GRADIENT CHECK HEATMAP
# ─────────────────────────────────────────────

def plot_gradient_check(
    manual_grad: torch.Tensor,
    auto_grad: torch.Tensor,
    title: str = "Manual vs Autograd Gradient"
) -> plt.Figure:
    """
    Side-by-side heatmap of manual vs autograd gradients + absolute error.
    Useful for verifying correctness of hand-derived backward passes.
    """
    manual_np = manual_grad.detach().numpy()
    auto_np   = auto_grad.detach().numpy()
    diff_np   = np.abs(manual_np - auto_np)

    # if 1D, reshape to 2D for heatmap
    if manual_np.ndim == 1:
        manual_np = manual_np[None, :]
        auto_np   = auto_np[None, :]
        diff_np   = diff_np[None, :]

    vmax = max(np.abs(manual_np).max(), np.abs(auto_np).max())

    fig, axes = plt.subplots(1, 3, figsize=(13, max(3, manual_np.shape[0] * 0.5 + 2)))

    im0 = axes[0].imshow(manual_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0].set_title("Manual backward", fontweight='bold')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(auto_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[1].set_title("PyTorch autograd", fontweight='bold')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(diff_np, cmap='Oranges', vmin=0, aspect='auto')
    axes[2].set_title("Absolute error", fontweight='bold', color='#C44E52')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(title, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# NORMALIZATION GEOMETRY
# ─────────────────────────────────────────────

def plot_normalization_geometry(n_samples: int = 6, n_features: int = 8) -> plt.Figure:
    """
    Visual diagram showing WHICH dimensions each norm type reduces over.
    Renders a (N x D) grid where colored cells = "this cell is averaged over".

    Intuition builder: helps explain why BatchNorm fails with small batches
    and why LayerNorm is preferred in transformers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ['BatchNorm\n(normalize over N per feature)', 
              'LayerNorm\n(normalize over D per sample)',
              'RMSNorm\n(normalize over D, no mean shift)']
    colors = [PALETTE['batchnorm'], PALETTE['layernorm'], PALETTE['rmsnorm']]

    for ax, title, color in zip(axes, titles, colors):
        grid = np.zeros((n_samples, n_features))

        if 'Batch' in title:
            # highlight one column (one feature, all samples)
            grid[:, 2] = 1
            label = "↕ reduces over batch (N)\nfor each feature independently"
        else:
            # highlight one row (one sample, all features)
            grid[2, :] = 1
            label = "→ reduces over features (D)\nfor each sample independently"

        ax.imshow(grid, cmap=plt.cm.colors.ListedColormap(['#EEEEEE', color]),
                  vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_samples))
        ax.set_xticklabels([f'f{i}' for i in range(n_features)], fontsize=7)
        ax.set_yticklabels([f's{i}' for i in range(n_samples)], fontsize=7)
        ax.set_xlabel("Feature dimension (D)", fontsize=9)
        ax.set_ylabel("Sample dimension (N)", fontsize=9)
        ax.set_title(title, fontweight='bold', color=color, fontsize=10)

        # add text annotation
        ax.text(0.5, -0.22, label, transform=ax.transAxes,
                ha='center', fontsize=8, color='#444444')

    fig.suptitle("Normalization Geometry: which dimensions are averaged?",
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# BATCHNORM RUNNING STATS
# ─────────────────────────────────────────────

def plot_running_stats(
    running_means: np.ndarray,
    running_vars: np.ndarray,
    true_means: np.ndarray,
    true_vars: np.ndarray,
    feature_idx: int = 0,
) -> plt.Figure:
    """
    Show how running_mean and running_var converge to true population stats.
    """
    steps = np.arange(len(running_means))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(steps, running_means[:, feature_idx],
             color=PALETTE['batchnorm'], linewidth=2, label='running_mean (EMA)')
    ax1.axhline(true_means[feature_idx], color='black', linestyle='--',
                linewidth=1.5, label=f'true mean = {true_means[feature_idx]:.2f}')
    ax1.set_title("Running Mean convergence", fontweight='bold')
    ax1.set_xlabel("Training step"); ax1.set_ylabel("Mean")
    ax1.legend()

    ax2.plot(steps, running_vars[:, feature_idx],
             color=PALETTE['batchnorm'], linewidth=2, label='running_var (EMA)')
    ax2.axhline(true_vars[feature_idx], color='black', linestyle='--',
                linewidth=1.5, label=f'true var = {true_vars[feature_idx]:.2f}')
    ax2.set_title("Running Variance convergence", fontweight='bold')
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Variance")
    ax2.legend()

    fig.tight_layout()
    return fig
