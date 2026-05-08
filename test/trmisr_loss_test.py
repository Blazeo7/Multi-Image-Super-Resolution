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
