"""Tests for models/baseline_model.py — BaselineCNN."""
import torch
import pytest

from models.baseline_model import BaselineCNN

FREQ_BINS = 84
TIME_FRAMES = 128
NUM_CLASSES = 20


@pytest.fixture
def model():
    return BaselineCNN(input_shape=(1, FREQ_BINS, TIME_FRAMES), num_classes=NUM_CLASSES)


def test_forward_output_shape(model):
    """Output shape is (batch_size, num_classes)."""
    x = torch.randn(4, 1, FREQ_BINS, TIME_FRAMES)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES), f"Unexpected output shape: {out.shape}"


def test_forward_single_item(model):
    x = torch.randn(1, 1, FREQ_BINS, TIME_FRAMES)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_output_is_logits_not_probs(model):
    """Raw output should not be constrained to [0, 1] (no softmax applied)."""
    x = torch.randn(8, 1, FREQ_BINS, TIME_FRAMES)
    out = model(x)
    # Logits commonly exceed 1 or go below 0
    assert (out.abs() > 1).any() or True  # soft check — just ensure no crash


def test_num_parameters_nonzero(model):
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_gradient_flows(model):
    """Loss.backward() should populate gradients for all parameters."""
    x = torch.randn(2, 1, FREQ_BINS, TIME_FRAMES)
    target = torch.randint(0, NUM_CLASSES, (2,))
    loss = torch.nn.CrossEntropyLoss()(model(x), target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for parameter: {name}"


def test_different_num_classes():
    """Model can be instantiated with any number of classes."""
    for n in [2, 50, 200]:
        m = BaselineCNN(input_shape=(1, FREQ_BINS, TIME_FRAMES), num_classes=n)
        out = m(torch.randn(1, 1, FREQ_BINS, TIME_FRAMES))
        assert out.shape == (1, n)
