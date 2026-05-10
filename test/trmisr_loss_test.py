import pytest
import torch
from losses.trmisr_loss import TRMISRLoss

# --- Fixtures ---


@pytest.fixture
def loss_fn():
    """Provides a TRMISRLoss instance with a fixed max_shift."""
    return TRMISRLoss(max_shift=2, loss_type="l2")


@pytest.fixture
def dummy_hr():
    """Creates a random 32x32 HR image for testing."""
    torch.manual_seed(42)  # For reproducibility
    return torch.rand(1, 1, 32, 32)


@pytest.fixture
def dummy_ycbcr():
    """Creates a random 3-channel (YCbCr) 32x32 image for testing."""
    torch.manual_seed(42)  # For reproducibility
    return torch.rand(1, 3, 32, 32)


# --- Tests ---


def test_perfect_match(loss_fn, dummy_hr):
    """Test 1: Loss should be zero for an identical image."""
    loss = loss_fn(dummy_hr, dummy_hr)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("offset", [0.1, 0.5, -0.3])
def test_brightness_invariance(loss_fn, dummy_hr, offset):
    """Test 2: Loss should be zero regardless of global brightness shifts."""
    sr_bright = dummy_hr + offset
    loss = loss_fn(sr_bright, dummy_hr)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("shift_x, shift_y", [(1, 0), (0, 1), (2, 2), (-1, 0), (0, -2)])
def test_shift_invariance(loss_fn, dummy_hr, shift_x, shift_y):
    """Test 3: Loss should be zero if within max_shift (c=2)."""
    sr_shifted = torch.roll(dummy_hr, shifts=(shift_x, shift_y), dims=(3, 2))

    loss = loss_fn(sr_shifted, dummy_hr)

    # Debugging print (visible if you run pytest -s)
    print(f"Testing shift ({shift_x}, {shift_y}), Loss: {loss.item()}")

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_combined_invariance(loss_fn, dummy_hr):
    """Test 4: Both shift and brightness at the same time."""
    sr = torch.zeros_like(dummy_hr)
    sr[:, :, 1:, 2:] = dummy_hr[:, :, :-1, :-2]  # Shift
    sr += 0.25  # Brightness

    loss = loss_fn(sr, dummy_hr)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_real_error(loss_fn, dummy_hr):
    """Test 5: Random noise should result in a significant loss value."""
    sr_noisy = dummy_hr + torch.randn_like(dummy_hr) * 0.2
    loss = loss_fn(sr_noisy, dummy_hr)
    assert loss.item() > 1e-3


def test_gradients(loss_fn, dummy_hr):
    """Test 6: Ensure the loss is differentiable (Autograd works)."""
    sr = dummy_hr.clone().detach().requires_grad_(True)
    loss = loss_fn(sr, dummy_hr)
    loss.backward()

    assert sr.grad is not None
    assert not torch.isnan(sr.grad).any()


def test_ycbcr_independent_channel_brightness_invariance(loss_fn, dummy_ycbcr):
    """
    Test 7: Loss should be zero when different brightness offsets are applied
    independently to the Y, Cb, and Cr channels. This proves the brightness
    correction is calculated per-channel, not globally across all colors.
    """
    sr_ycbcr = dummy_ycbcr.clone()

    # Apply completely different brightness offsets to each channel
    sr_ycbcr[:, 0, :, :] += 0.5  # Y channel offset
    sr_ycbcr[:, 1, :, :] -= 0.3  # Cb channel offset
    sr_ycbcr[:, 2, :, :] += 0.15  # Cr channel offset

    loss = loss_fn(sr_ycbcr, dummy_ycbcr)

    # If dim=(1,2,3) was mistakenly used in the loss function,
    # this assert would fail because the channels would cross-contaminate.
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_ycbcr_y_channel_only(dummy_ycbcr):
    """
    Test 8: Verify that setting y_channel_only=True ignores errors in Cb and Cr.
    """
    # Create a new loss function instance with the flag enabled
    loss_fn_y_only = TRMISRLoss(max_shift=2, loss_type="l2", y_channel_only=True)

    sr_ycbcr = dummy_ycbcr.clone()

    # Corrupt the Cb and Cr channels with random noise
    sr_ycbcr[:, 1:, :, :] += torch.randn_like(sr_ycbcr[:, 1:, :, :])

    # Because the Y channel (index 0) is perfectly identical, the loss
    # should be 0 regardless of how messed up the color channels are.
    loss = loss_fn_y_only(sr_ycbcr, dummy_ycbcr)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
