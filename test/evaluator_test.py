import pytest
import torch
from evaluator.metric_tracker import compute_cpsnr, compute_cssim


@pytest.fixture
def dummy_data():
    """
    Generates a random HR image.
    Note: We use a fixed seed to ensure deterministic tests.
    """
    torch.manual_seed(42)
    hr = torch.rand(1, 1, 64, 64)  # (B, C, H, W)
    return hr


@pytest.fixture
def max_off():
    return 3


def test_cpsnr_perfect_match(dummy_data, max_off):
    """If images are identical, cPSNR should be very high (capped by 1e-10 MSE)."""
    val = compute_cpsnr(dummy_data, dummy_data, max_offset=max_off)
    # With 1-bit depth and 1e-10 MSE clamp, max cPSNR is ~100 dB
    assert val == pytest.approx(100.0, abs=1e-7)


def test_cpsnr_brightness_invariance(dummy_data, max_off):
    """Adding a constant brightness should not change the cPSNR (it should stay high)."""
    sr = dummy_data + 0.2
    val = compute_cpsnr(sr, dummy_data, max_offset=max_off)
    assert val == pytest.approx(100.0, abs=1e-7)


@pytest.mark.parametrize("shift_x, shift_y", [(1, 2), (-2, 1), (3, 0), (0, -3)])
def test_cpsnr_shift_invariance(dummy_data, max_off, shift_x, shift_y):
    """Shifting within max_offset should still result in a perfect match cPSNR."""
    # sr is a shifted version of hr
    sr = torch.roll(dummy_data, shifts=(shift_x, shift_y), dims=(3, 2))
    val = compute_cpsnr(sr, dummy_data, max_offset=max_off)
    assert val == pytest.approx(100.0, abs=1e-7)


def test_cssim_perfect_match(dummy_data, max_off):
    """Identical images must have SSIM of 1.0."""
    val = compute_cssim(dummy_data, dummy_data, max_offset=max_off)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_cssim_brightness_invariance(dummy_data, max_off):
    offset = 0.05
    sr = dummy_data + offset  # Do not clamp
    val = compute_cssim(sr, dummy_data, max_offset=max_off)
    assert val == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("shift_x, shift_y", [(2, 2), (-1, -1)])
def test_cssim_shift_invariance(dummy_data, max_off, shift_x, shift_y):
    """SSIM should be 1.0 if the shift is within the search range."""
    sr = torch.roll(dummy_data, shifts=(shift_x, shift_y), dims=(3, 2))
    val = compute_cssim(sr, dummy_data, max_offset=max_off)
    assert val == pytest.approx(1.0, abs=1e-4)


def test_cssim_mismatch(dummy_data, max_off):
    """Random noise should significantly lower the SSIM."""
    sr_noisy = torch.rand_like(dummy_data)
    val = compute_cssim(sr_noisy, dummy_data, max_offset=max_off)
    assert val < 0.5  # Random vs Random usually has very low SSIM
