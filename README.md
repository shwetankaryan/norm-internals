# Normalization Internals

**Building BatchNorm, LayerNorm, and RMSNorm from scratch — forward pass, manual backward, and training experiments.**

This project is in the spirit of [Karpathy's micrograd/makemore](https://github.com/karpathy/micrograd): every abstraction is broken open, every gradient is derived by hand, and every claim is verified experimentally.

---

## What's here

| Notebook | What you'll build |
|---|---|
| `01_batchnorm_internals.ipynb` | BatchNorm forward + manual backward, running stats, small-batch failure |
| `02_layernorm_internals.ipynb` | LayerNorm forward + manual backward, transformer use case, batch-invariance proof |
| `03_rmsnorm_internals.ipynb` | RMSNorm (used in LLaMA/Mistral), cost comparison, scale-equivariance |
| `04_full_comparison.ipynb` | All three side-by-side: gradient flow, Pre-Norm vs Post-Norm residual, init sensitivity |

---

## View Notebooks
- [01 BatchNorm Internals] (https://nbviewer.org/github/shwetankaryan/norm-internals/blob/main/notebooks/01_batchnorm_internals.ipynb)
- [02 LayerNorm Internals] (https://nbviewer.org/github/shwetankaryan/norm-internals/blob/main/notebooks/02_layernorm_internals.ipynb)
- [03 RMSNorm Internals] (https://nbviewer.org/github/shwetankaryan/norm-internals/blob/main/notebooks/03_rmsnorm_internals.ipynb)
- [04 Full Comparison] (https://nbviewer.org/github/shwetankaryan/norm-internals/blob/main/notebooks/04_full_comparison.ipynb)

## The core idea

Most engineers know how to *use* `nn.BatchNorm1d`. This project is about understanding what happens when you call `.backward()` on it.

The backward pass through BatchNorm is non-trivial because `x` affects the output through **three paths**:
1. Directly through `x_hat = (x - mu) / std`
2. Indirectly through `mu = mean(x)`
3. Indirectly through `var = mean((x - mu)^2)`

Each path contributes to `dx`, and getting the sum right requires careful chain rule bookkeeping. We derive it from scratch, implement it in ~20 lines of PyTorch, and verify against autograd with a gradient check.

---

## Key results

**Gradient check (manual vs autograd):**
```
BatchNorm:  dx ✓  d_gamma ✓  d_beta ✓   (max diff < 1e-7)
LayerNorm:  dx ✓  d_gamma ✓  d_beta ✓   (max diff < 1e-7)
RMSNorm:    dx ✓  d_gamma ✓              (max diff < 1e-7)
```

**The small-batch failure of BatchNorm** — visible empirically with batch sizes 2, 4, 8 vs 64.

**Pre-Norm vs Post-Norm residuals** — Pre-Norm (modern default in GPT-2/LLaMA) trains faster due to unobstructed residual gradient path.

**RMSNorm speedup** — ~7% faster per forward pass on CPU at d_model=4096, compounding at scale.

---

## Structure

```
norm-internals/
├── notebooks/
│   ├── 01_batchnorm_internals.ipynb
│   ├── 02_layernorm_internals.ipynb
│   ├── 03_rmsnorm_internals.ipynb
│   └── 04_full_comparison.ipynb
├── src/
│   ├── norms.py              # Manual forward + backward for all 3 norm types
│   ├── visualize.py          # Plotting utilities (gradient flow, distributions, etc.)
│   ├── train.py              # MLP training harness with swappable norm
│   └── generate_notebooks.py # Regenerate notebooks from source
├── tests/
│   └── test_norms.py         # Gradient checks + property tests
└── requirements.txt
```

---

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_batchnorm_internals.ipynb
```

To run tests:
```bash
pytest tests/ -v
```

To regenerate notebooks from source (if you modify `generate_notebooks.py`):
```bash
python src/generate_notebooks.py
```

---

## The math, in one place

### BatchNorm backward (5-step derivation)

Given upstream gradient `dout = dL/d(out)`:

```
d_gamma = sum(dout * x_hat,   dim=0)          # (D,)
d_beta  = sum(dout,           dim=0)          # (D,)
d_x_hat = dout * gamma                        # (N, D)
d_var   = sum(d_x_hat * (x-mu) * -0.5 * (var+eps)^(-3/2), dim=0)   # (D,)
d_mu    = sum(-d_x_hat / std, dim=0) + d_var * mean(-2*(x-mu))      # (D,)
dx      = d_x_hat/std + d_var * 2*(x-mu)/N + d_mu/N                 # (N, D)
```

### LayerNorm backward

Structurally identical, but `sum` axes swap: `dim=0` → `dim=-1`.

### RMSNorm backward (projection form)

```
d_gamma = sum(dout * x_hat,  dim=0)
d_x_hat = dout * gamma
proj    = sum(d_x_hat * x_hat, dim=-1, keepdim=True) / D
dx      = (d_x_hat - x_hat * proj) / rms
```

The `proj` term is the component of `d_x_hat` that lies along `x_hat` — it gets subtracted because that direction is "absorbed" by the RMS normalization.

---

## References

- Ioffe & Szegedy (2015) — [Batch Normalization](https://arxiv.org/abs/1502.03167)
- Ba et al. (2016) — [Layer Normalization](https://arxiv.org/abs/1607.06450)
- Zhang & Sennrich (2019) — [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Xiong et al. (2020) — [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (Pre-Norm vs Post-Norm)
- Touvron et al. (2023) — [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (RMSNorm in practice)
