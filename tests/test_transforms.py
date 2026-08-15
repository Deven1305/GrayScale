import pytest
import torch
import torch.nn.functional as F

from src.models.nafnet import NAFNetSR

def test_log_mode_baseline():
    """Verify log mode reproduces the exact baseline behavior."""
    model = NAFNetSR(in_ch=1, input_transform="log", use_log_channel=True)
    x = torch.tensor([[[[0.0, 1e-4], [0.5, 1.0]]]], dtype=torch.float32)
    stem_out = model._stem(x)
    
    # Check shape is (1, 2, 2, 2)
    assert stem_out.shape == (1, 2, 2, 2)
    
    # Original signal remains unchanged
    assert torch.allclose(stem_out[:, 0:1], x)
    
    # Check log values
    expected_log = torch.log(torch.tensor([1e-3, 1e-3, 0.5, 1.0]))
    assert torch.allclose(stem_out[0, 1].flatten(), expected_log)


def test_asinh_mode_handles_negative():
    """Verify asinh mode handles negative inputs without NaN or Inf."""
    model = NAFNetSR(in_ch=1, input_transform="asinh", use_log_channel=True)
    x = torch.tensor([[[[-0.1, -0.05], [0.0, 0.5]]]], dtype=torch.float32)
    stem_out = model._stem(x)
    
    # Check shape
    assert stem_out.shape == (1, 2, 2, 2)
    
    # Original signal remains unchanged
    assert torch.allclose(stem_out[:, 0:1], x)
    
    # Check asinh values: asinh(x / 0.1)
    expected_asinh = torch.arcsinh(x.flatten() / 0.1)
    assert torch.allclose(stem_out[0, 1].flatten(), expected_asinh)
    assert not torch.isnan(stem_out).any()
    assert not torch.isinf(stem_out).any()


def test_transform_is_differentiable():
    """Verify the transform is fully differentiable and does not break the graph."""
    model_log = NAFNetSR(in_ch=1, input_transform="log", use_log_channel=True)
    model_asinh = NAFNetSR(in_ch=1, input_transform="asinh", use_log_channel=True)
    
    x1 = torch.rand(1, 1, 16, 16, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)
    
    out_log = model_log._stem(x1)
    out_log.sum().backward()
    assert x1.grad is not None
    assert not torch.isnan(x1.grad).any()
    
    out_asinh = model_asinh._stem(x2)
    out_asinh.sum().backward()
    assert x2.grad is not None
    assert not torch.isnan(x2.grad).any()


def test_missing_field_defaults_to_log():
    """An old checkpoint without input_transform defaults to log."""
    model = NAFNetSR(in_ch=1, use_log_channel=True) # no input_transform passed
    assert getattr(model, "input_transform", "log") == "log"
    
    x = torch.tensor([[[[0.0]]]])
    stem_out = model._stem(x)
    assert torch.allclose(stem_out[0, 1], torch.tensor(torch.log(torch.tensor(1e-3))))
