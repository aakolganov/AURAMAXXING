"""MACEInterface float32-safe load fallback (for the MPS float64 limitation).

These exercise the fallback logic only, injecting a fake ``mace_mp`` so they need
neither a real MACE install nor an MPS device (torch is still required, so they
skip where it is absent).
"""
import pytest


def test_resolve_dtype_defaults_by_device():
    from interfaces.MACE_interface import MACEInterface
    # float64 on CPU/CUDA (accurate, recommended for opt), float32 on MPS (no float64).
    assert MACEInterface._resolve_dtype("cpu", None) == "float64"
    assert MACEInterface._resolve_dtype("cuda", None) == "float64"
    assert MACEInterface._resolve_dtype("mps", None) == "float32"


def test_resolve_dtype_explicit_wins():
    from interfaces.MACE_interface import MACEInterface
    assert MACEInterface._resolve_dtype("cpu", "float32") == "float32"
    assert MACEInterface._resolve_dtype("mps", "float64") == "float64"


def test_float32_fallback_on_mps_load_error(tmp_path):
    torch = pytest.importorskip("torch")
    from interfaces.MACE_interface import MACEInterface

    # A tiny float64 "checkpoint" standing in for a MACE model.
    model_path = tmp_path / "fake.model"
    torch.save(torch.nn.Linear(2, 2).double(), model_path)

    calls = []
    sentinel = object()

    def fake_mace_mp(model, **kwargs):
        calls.append(model)
        if len(calls) == 1:
            raise TypeError("Cannot convert a MPS Tensor to float64 dtype (MPS)")
        return sentinel

    result = MACEInterface._load_mace_mp(fake_mace_mp, str(model_path), {"device": "mps"})

    assert result is sentinel
    assert len(calls) == 2                      # first attempt failed, retried
    assert calls[1] != str(model_path)          # retry used the converted float32 file


def test_no_fallback_when_direct_load_succeeds():
    pytest.importorskip("torch")
    from interfaces.MACE_interface import MACEInterface

    calls = []
    sentinel = object()

    def fake_mace_mp(model, **kwargs):
        calls.append(model)
        return sentinel

    result = MACEInterface._load_mace_mp(fake_mace_mp, "model.model", {"device": "cpu"})
    assert result is sentinel
    assert len(calls) == 1                       # no retry


def test_unrelated_typeerror_is_reraised():
    pytest.importorskip("torch")
    from interfaces.MACE_interface import MACEInterface

    def fake_mace_mp(model, **kwargs):
        raise TypeError("some unrelated problem")

    with pytest.raises(TypeError, match="unrelated"):
        MACEInterface._load_mace_mp(fake_mace_mp, "model.model", {"device": "mps"})
