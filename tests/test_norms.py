"""
tests/test_norms.py — Unit tests for manual normalization implementations.

Run with: python -m pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from norms import BatchNorm1dManual, LayerNormManual, RMSNormManual, check_gradients


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def small_input():
    torch.manual_seed(42)
    return torch.randn(8, 16)

@pytest.fixture
def medium_input():
    torch.manual_seed(0)
    return torch.randn(32, 64)


# ─────────────────────────────────────────────
# BATCHNORM TESTS
# ─────────────────────────────────────────────

class TestBatchNorm1dManual:

    def test_output_shape(self, small_input):
        bn = BatchNorm1dManual(num_features=16)
        out = bn.forward(small_input)
        assert out.shape == small_input.shape

    def test_normalized_mean(self, small_input):
        """Output per-feature mean should be ~0 (gamma=1, beta=0)"""
        bn = BatchNorm1dManual(num_features=16)
        out = bn.forward(small_input)
        assert out.mean(dim=0).abs().max() < 1e-5, "Per-feature mean should be ~0"

    def test_normalized_std(self, small_input):
        """Output per-feature std should be ~1"""
        bn = BatchNorm1dManual(num_features=16)
        out = bn.forward(small_input)
        # var() uses unbiased=True by default, but we normalized with biased var
        # so we check std is close to 1, not exactly 1
        assert (out.std(dim=0) - 1.0).abs().max() < 0.1

    def test_affine_transform(self, small_input):
        """With gamma=2, beta=3 output should be 2*x_hat + 3"""
        bn = BatchNorm1dManual(num_features=16)
        bn.gamma = torch.full((16,), 2.0)
        bn.beta  = torch.full((16,), 3.0)
        out = bn.forward(small_input)
        assert (out.mean(dim=0) - 3.0).abs().max() < 1e-4

    def test_running_stats_update(self, small_input):
        """Running stats should change after a forward pass"""
        bn = BatchNorm1dManual(num_features=16, momentum=0.1)
        initial_mean = bn.running_mean.clone()
        bn.forward(small_input)
        assert not torch.allclose(bn.running_mean, initial_mean)

    def test_backward_gradient_check(self, small_input):
        bn = BatchNorm1dManual(num_features=16)
        results = check_gradients(bn, small_input)
        assert results['all_pass'], f"Gradient check failed: {results}"

    def test_backward_gradient_check_medium(self, medium_input):
        bn = BatchNorm1dManual(num_features=64)
        results = check_gradients(bn, medium_input)
        assert results['all_pass'], f"Gradient check failed on medium input: {results}"

    def test_matches_pytorch(self, small_input):
        """Manual forward should match nn.BatchNorm1d output"""
        N, D = small_input.shape
        bn_manual = BatchNorm1dManual(num_features=D)
        bn_torch  = nn.BatchNorm1d(D, eps=bn_manual.eps, momentum=bn_manual.momentum)
        bn_torch.weight.data = bn_manual.gamma.clone()
        bn_torch.bias.data   = bn_manual.beta.clone()
        bn_torch.train()

        out_manual = bn_manual.forward(small_input)
        out_torch  = bn_torch(small_input)
        assert torch.allclose(out_manual, out_torch, atol=1e-5), \
            f"Max diff: {(out_manual - out_torch).abs().max()}"

    def test_eval_mode_uses_running_stats(self):
        """At eval time, output should use running_mean/var not batch stats"""
        torch.manual_seed(1)
        D = 8
        bn = BatchNorm1dManual(num_features=D, momentum=0.5)

        # set running stats to known values
        bn.running_mean = torch.ones(D) * 2.0
        bn.running_var  = torch.ones(D) * 4.0
        bn.training = False

        x = torch.zeros(4, D)
        out = bn.forward(x)

        # x=0, running_mean=2, running_var=4 → x_hat = (0-2)/sqrt(4+eps) ≈ -1
        expected = -2.0 / (4.0 + bn.eps) ** 0.5
        assert (out - expected).abs().max() < 1e-4


# ─────────────────────────────────────────────
# LAYERNORM TESTS
# ─────────────────────────────────────────────

class TestLayerNormManual:

    def test_output_shape(self, small_input):
        ln = LayerNormManual(normalized_shape=16)
        out = ln.forward(small_input)
        assert out.shape == small_input.shape

    def test_per_sample_mean_zero(self, small_input):
        """Each sample's output should have mean ~0"""
        ln = LayerNormManual(normalized_shape=16)
        out = ln.forward(small_input)
        assert out.mean(dim=-1).abs().max() < 1e-5

    def test_per_sample_std_one(self, small_input):
        """Each sample's output should have std ~1"""
        ln = LayerNormManual(normalized_shape=16)
        out = ln.forward(small_input)
        assert (out.std(dim=-1) - 1.0).abs().max() < 0.1

    def test_batch_size_one(self):
        """LayerNorm must work with batch size = 1"""
        x = torch.randn(1, 32)
        ln = LayerNormManual(normalized_shape=32)
        out = ln.forward(x)
        assert out.shape == x.shape
        assert out.mean(dim=-1).abs().max() < 1e-5

    def test_backward_gradient_check(self, small_input):
        ln = LayerNormManual(normalized_shape=16)
        results = check_gradients(ln, small_input)
        assert results['all_pass'], f"Gradient check failed: {results}"

    def test_matches_pytorch(self, medium_input):
        N, D = medium_input.shape
        ln_manual = LayerNormManual(normalized_shape=D)
        ln_torch  = nn.LayerNorm(D, eps=ln_manual.eps)
        ln_torch.weight.data = ln_manual.gamma.clone()
        ln_torch.bias.data   = ln_manual.beta.clone()

        out_manual = ln_manual.forward(medium_input)
        out_torch  = ln_torch(medium_input)
        assert torch.allclose(out_manual, out_torch, atol=1e-5), \
            f"Max diff: {(out_manual - out_torch).abs().max()}"

    def test_independent_of_batch(self):
        """Adding or removing samples should not change existing outputs"""
        torch.manual_seed(3)
        x1 = torch.randn(1, 16)
        x2 = torch.randn(1, 16)
        x12 = torch.cat([x1, x2], dim=0)

        ln = LayerNormManual(normalized_shape=16)
        out1  = ln.forward(x1)
        out12 = ln.forward(x12)

        assert torch.allclose(out1[0], out12[0], atol=1e-6), \
            "LayerNorm output for x1 should be same whether processed alone or with x2"


# ─────────────────────────────────────────────
# RMSNORM TESTS
# ─────────────────────────────────────────────

class TestRMSNormManual:

    def test_output_shape(self, small_input):
        rms = RMSNormManual(normalized_shape=16)
        out = rms.forward(small_input)
        assert out.shape == small_input.shape

    def test_output_rms_is_one(self, small_input):
        """Output RMS should be ~1 (when gamma=1)"""
        rms = RMSNormManual(normalized_shape=16)
        out = rms.forward(small_input)
        rms_out = torch.sqrt((out**2).mean(dim=-1))
        assert (rms_out - 1.0).abs().max() < 1e-5

    def test_no_mean_centering(self):
        """RMSNorm should NOT zero the mean"""
        x = torch.ones(4, 8) * 5.0   # all-positive input
        rms = RMSNormManual(normalized_shape=8)
        out = rms.forward(x)
        # all-ones input: RMS = 1, so output = gamma * (x / RMS) = gamma * 1 = 1
        # mean is still positive
        assert out.mean().item() > 0, "RMSNorm should not center the mean"

    def test_no_beta_parameter(self):
        rms = RMSNormManual(normalized_shape=16)
        assert len(rms.parameters()) == 1, "RMSNorm should have only gamma, no beta"

    def test_backward_gradient_check(self, small_input):
        rms = RMSNormManual(normalized_shape=16)
        results = check_gradients(rms, small_input)
        assert results['all_pass'], f"Gradient check failed: {results}"

    def test_backward_gradient_check_medium(self, medium_input):
        rms = RMSNormManual(normalized_shape=64)
        results = check_gradients(rms, medium_input)
        assert results['all_pass'], f"Gradient check failed on medium: {results}"

    def test_matches_manual_computation(self, small_input):
        """Verify forward output manually"""
        D = 16
        rms_norm = RMSNormManual(normalized_shape=D)
        out = rms_norm.forward(small_input)

        rms = torch.sqrt((small_input**2).mean(dim=-1, keepdim=True) + rms_norm.eps)
        expected = rms_norm.gamma * (small_input / rms)
        assert torch.allclose(out, expected, atol=1e-6)


# ─────────────────────────────────────────────
# CROSS-NORM PROPERTIES
# ─────────────────────────────────────────────

class TestCrossNormProperties:

    def test_batchnorm_fails_batchsize_one(self):
        """BatchNorm variance is 0 for batch size 1 during training"""
        x = torch.randn(1, 8)
        bn = BatchNorm1dManual(num_features=8)
        out = bn.forward(x)
        # var = 0, so x_hat = 0/eps = very small or nan
        # just check it doesn't crash and output has right shape
        assert out.shape == x.shape

    def test_layernorm_batch_invariant(self):
        """Processing samples one-by-one vs in batch must give same output"""
        torch.manual_seed(5)
        xs = [torch.randn(1, 16) for _ in range(4)]
        x_batch = torch.cat(xs, dim=0)

        ln = LayerNormManual(normalized_shape=16)
        out_batch = ln.forward(x_batch)

        for i, xi in enumerate(xs):
            out_i = ln.forward(xi)
            assert torch.allclose(out_i[0], out_batch[i], atol=1e-6)

    def test_rmsnorm_scale_equivariance(self):
        """RMSNorm(c * x) = c/|c| * RMSNorm(x) when c is scalar != 0"""
        torch.manual_seed(6)
        x   = torch.randn(4, 16)
        c   = 3.0
        rms = RMSNormManual(normalized_shape=16)

        out_x  = rms.forward(x)
        out_cx = rms.forward(c * x)
        # Since RMS(c*x) = |c| * RMS(x), the normalization cancels the scale
        assert torch.allclose(out_x, out_cx, atol=1e-5), \
            "RMSNorm should be scale-equivariant"
