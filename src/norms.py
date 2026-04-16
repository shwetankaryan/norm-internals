"""
norms.py — Manual forward + backward implementations of normalization layers.

Every backward pass here is derived by hand from the chain rule.
No autograd is used in the *_backward functions.
We verify correctness by comparing gradients against PyTorch's autograd.
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# BATCH NORMALIZATION
# ─────────────────────────────────────────────

class BatchNorm1dManual:
    """
    BatchNorm over (N, D) input — normalizes across the N (batch) dimension.

    During training:
        mu    = mean over batch
        var   = variance over batch (biased, i.e. /N not /N-1)
        x_hat = (x - mu) / sqrt(var + eps)
        out   = gamma * x_hat + beta

    During inference:
        uses running_mean, running_var accumulated during training.

    We cache intermediate tensors in forward() that are needed by backward().
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # learnable affine parameters
        self.gamma = torch.ones(num_features)   # scale
        self.beta  = torch.zeros(num_features)  # shift

        # running stats (updated during training, used during inference)
        self.running_mean = torch.zeros(num_features)
        self.running_var  = torch.ones(num_features)

        self.training = True

        # cache for backward
        self._cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D)"""
        assert x.ndim == 2, "BatchNorm1dManual expects (N, D) input"
        N, D = x.shape

        if self.training:
            mu  = x.mean(dim=0)                          # (D,)
            var = x.var(dim=0, unbiased=False)            # (D,) biased variance

            x_hat = (x - mu) / torch.sqrt(var + self.eps) # (N, D)
            out   = self.gamma * x_hat + self.beta         # (N, D)

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var

            # cache everything needed for backward
            self._cache = {
                'x': x, 'x_hat': x_hat,
                'mu': mu, 'var': var,
                'N': N, 'D': D,
            }
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            out   = self.gamma * x_hat + self.beta

        return out

    def backward(self, dout: torch.Tensor):
        """
        Manual backward through BatchNorm.

        Given dL/d(out), compute:
          dL/d(gamma), dL/d(beta), dL/d(x)

        Derivation (using the chain rule step-by-step):

          out   = gamma * x_hat + beta
          x_hat = (x - mu) / std          where std = sqrt(var + eps)
          mu    = mean(x)
          var   = mean((x - mu)^2)

        Step 1: gradients w.r.t. affine params
          d_gamma = sum(dout * x_hat, dim=0)
          d_beta  = sum(dout, dim=0)

        Step 2: gradient w.r.t. x_hat
          d_x_hat = dout * gamma           # (N, D)

        Step 3: gradient w.r.t. variance
          d_var = sum(d_x_hat * (x - mu) * -0.5 * (var + eps)^(-3/2), dim=0)

        Step 4: gradient w.r.t. mean
          d_mu = sum(-d_x_hat / std, dim=0) + d_var * mean(-2*(x - mu))

        Step 5: gradient w.r.t. x
          dx = d_x_hat / std + d_var * 2*(x - mu)/N + d_mu/N
        """
        x, x_hat = self._cache['x'], self._cache['x_hat']
        mu, var  = self._cache['mu'], self._cache['var']
        N        = self._cache['N']

        std     = torch.sqrt(var + self.eps)   # (D,)
        x_mu    = x - mu                       # (N, D)

        # affine grads
        d_gamma = (dout * x_hat).sum(dim=0)    # (D,)
        d_beta  = dout.sum(dim=0)              # (D,)

        # x_hat grad
        d_x_hat = dout * self.gamma            # (N, D)

        # variance grad
        d_var = (d_x_hat * x_mu * -0.5 * (var + self.eps).pow(-1.5)).sum(dim=0)  # (D,)

        # mean grad
        d_mu = (-d_x_hat / std).sum(dim=0) + d_var * (-2 * x_mu).mean(dim=0)    # (D,)

        # input grad
        dx = d_x_hat / std + d_var * 2 * x_mu / N + d_mu / N                    # (N, D)

        return dx, d_gamma, d_beta

    def parameters(self):
        return [self.gamma, self.beta]


# ─────────────────────────────────────────────
# LAYER NORMALIZATION
# ─────────────────────────────────────────────

class LayerNormManual:
    """
    LayerNorm over (N, D) input — normalizes across the D (feature) dimension.

    Key difference from BatchNorm:
      - Statistics are computed per sample, not per feature.
      - No running stats needed → works identically at train and inference time.
      - This is why transformers use LayerNorm: batch size = 1 works fine.

    out = gamma * (x - mu) / sqrt(var + eps) + beta
    where mu, var are computed over the last `normalized_shape` dimensions.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = torch.ones(normalized_shape)
        self.beta  = torch.zeros(normalized_shape)

        self._cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) — normalizes over D per sample"""
        mu  = x.mean(dim=-1, keepdim=True)               # (N, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # (N, 1)
        std = torch.sqrt(var + self.eps)                  # (N, 1)

        x_hat = (x - mu) / std                           # (N, D)
        out   = self.gamma * x_hat + self.beta            # (N, D)

        self._cache = {'x': x, 'x_hat': x_hat, 'mu': mu, 'var': var, 'std': std}
        return out

    def backward(self, dout: torch.Tensor):
        """
        Manual backward through LayerNorm.

        Same derivation structure as BatchNorm but reduction is over D (features)
        instead of N (batch). The N samples are independent here.

        d_gamma = sum(dout * x_hat, dim=0)       # sum over batch
        d_beta  = sum(dout, dim=0)
        d_x_hat = dout * gamma
        d_var   = sum(d_x_hat * (x-mu) * -0.5 * std^-3, dim=-1, keepdim=True)
        d_mu    = sum(-d_x_hat / std, dim=-1, keepdim=True) + d_var * mean(-2*(x-mu), dim=-1)
        dx      = d_x_hat/std + d_var * 2*(x-mu)/D + d_mu/D
        """
        x, x_hat = self._cache['x'], self._cache['x_hat']
        mu, var, std = self._cache['mu'], self._cache['var'], self._cache['std']
        N, D = x.shape
        x_mu = x - mu  # (N, D)

        d_gamma = (dout * x_hat).sum(dim=0)   # (D,)
        d_beta  = dout.sum(dim=0)             # (D,)

        d_x_hat = dout * self.gamma           # (N, D)

        d_var = (d_x_hat * x_mu * -0.5 * (var + self.eps).pow(-1.5)).sum(dim=-1, keepdim=True)  # (N,1)

        d_mu  = (-d_x_hat / std).sum(dim=-1, keepdim=True) + d_var * (-2 * x_mu).mean(dim=-1, keepdim=True)  # (N,1)

        dx = d_x_hat / std + d_var * 2 * x_mu / D + d_mu / D  # (N, D)

        return dx, d_gamma, d_beta

    def parameters(self):
        return [self.gamma, self.beta]


# ─────────────────────────────────────────────
# RMS NORMALIZATION
# ─────────────────────────────────────────────

class RMSNormManual(nn.Module):
    """
    RMSNorm — used in LLaMA, Mistral, Gemma instead of LayerNorm.

    Removes the mean-centering step entirely:
      rms   = sqrt(mean(x^2) + eps)
      x_hat = x / rms
      out   = gamma * x_hat       (no beta — no shift parameter)

    Why remove the mean?
      - Cheaper: one less reduction op.
      - Hypothesis: the re-centering in LayerNorm is redundant when
        the model has learned good weight initializations.
      - Empirically matches LayerNorm quality at lower cost.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = torch.ones(normalized_shape)
        self._cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D)"""
        rms   = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)  # (N, 1)
        x_hat = x / rms                                                       # (N, D)
        out   = self.gamma * x_hat                                            # (N, D)
        self._cache = {'x': x, 'x_hat': x_hat, 'rms': rms}
        return out

    def backward(self, dout: torch.Tensor):
        """
        Manual backward through RMSNorm.

        Let r = rms(x) = sqrt(mean(x^2) + eps)

        d_gamma = sum(dout * x_hat, dim=0)
        d_x_hat = dout * gamma                        # (N, D)

        Since x_hat = x / r:
          dx/dx_i: need to account for r depending on x

        dr/dx_i = x_i / (D * r)

        d(x_hat_j)/d(x_i):
          = (1/r) if i==j, else 0
          minus x_j/r^2 * dr/dx_i  (chain rule through r)
          = (1/r) * [delta_ij - x_j * x_i / (D * r^2)]

        Summing over j:
          dx_i = (1/r) * [d_x_hat_i - x_hat_i * sum(d_x_hat_j * x_hat_j) / D]
        """
        x, x_hat, rms = self._cache['x'], self._cache['x_hat'], self._cache['rms']
        N, D = x.shape

        d_gamma = (dout * x_hat).sum(dim=0)    # (D,)
        d_x_hat = dout * self.gamma            # (N, D)

        # contraction term: how much d_x_hat "projects" onto x_hat
        proj = (d_x_hat * x_hat).sum(dim=-1, keepdim=True) / D   # (N, 1)
        dx   = (d_x_hat - x_hat * proj) / rms                     # (N, D)

        return dx, d_gamma

    def parameters(self):
        return [self.gamma]


# ─────────────────────────────────────────────
# GRADIENT CHECKING UTILITIES
# ─────────────────────────────────────────────

def numerical_gradient(fn, tensor: torch.Tensor, h: float = 1e-4) -> torch.Tensor:
    """
    Compute numerical gradient via centered finite differences.
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    This is O(2 * numel(tensor)) forward passes — only for small tensors / testing.
    """
    grad = torch.zeros_like(tensor)
    it   = tensor.view(-1)
    for i in range(it.numel()):
        orig = it[i].item()
        it[i] = orig + h
        fwd   = fn().sum().item()
        it[i] = orig - h
        bwd   = fn().sum().item()
        it[i] = orig
        grad.view(-1)[i] = (fwd - bwd) / (2 * h)
    return grad


def check_gradients(layer, x: torch.Tensor, atol: float = 1e-4) -> dict:
    """
    Compare manual backward gradients against PyTorch autograd.

    Returns a dict of max absolute differences for each parameter.
    All values should be < atol for the implementation to be correct.
    """
    results = {}

    # ── autograd path ────────────────────────────────
    x_auto = x.clone().detach().requires_grad_(True)

    # rebuild a torch native equivalent
    D = x.shape[-1]
    if isinstance(layer, BatchNorm1dManual):
        torch_layer = nn.BatchNorm1d(D, eps=layer.eps, momentum=layer.momentum)
        torch_layer.weight.data = layer.gamma.clone()
        torch_layer.bias.data   = layer.beta.clone()
        torch_layer.train()
    elif isinstance(layer, LayerNormManual):
        torch_layer = nn.LayerNorm(D, eps=layer.eps)
        torch_layer.weight.data = layer.gamma.clone()
        torch_layer.bias.data   = layer.beta.clone()
    elif isinstance(layer, RMSNormManual):
        # PyTorch doesn't ship RMSNorm until 2.4 — use manual autograd version
        class _RMS(nn.Module):
            def __init__(self, g, eps):
                super().__init__()
                self.gamma = nn.Parameter(g.clone())
                self.eps   = eps
            def forward(self, x):
                rms = torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)
                return self.gamma * (x / rms)
        torch_layer = _RMS(layer.gamma, layer.eps)
    else:
        raise ValueError(f"Unknown layer type: {type(layer)}")

    out_auto = torch_layer(x_auto)
    loss_auto = out_auto.sum()
    loss_auto.backward()

    dout = torch.ones_like(out_auto)

    # ── manual backward path ─────────────────────────
    layer_params_before = [p.clone() for p in layer.parameters()]
    _ = layer.forward(x.clone().detach())
    grads = layer.backward(dout)

    # dx is always first
    dx_manual = grads[0]
    dx_auto   = x_auto.grad

    results['dx'] = (dx_manual - dx_auto).abs().max().item()

    # gamma grad
    if isinstance(layer, BatchNorm1dManual) or isinstance(layer, LayerNormManual):
        d_gamma_manual = grads[1]
        d_gamma_auto   = torch_layer.weight.grad
        results['d_gamma'] = (d_gamma_manual - d_gamma_auto).abs().max().item()

        d_beta_manual = grads[2]
        d_beta_auto   = torch_layer.bias.grad
        results['d_beta'] = (d_beta_manual - d_beta_auto).abs().max().item()

    elif isinstance(layer, RMSNormManual):
        d_gamma_manual = grads[1]
        d_gamma_auto   = torch_layer.gamma.grad
        results['d_gamma'] = (d_gamma_manual - d_gamma_auto).abs().max().item()

    results['all_pass'] = all(v < atol for v in results.values() if isinstance(v, float))
    return results
