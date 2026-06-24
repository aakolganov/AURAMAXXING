"""Pure-logic tests for the UMA interface decision helpers.

These exercise the parts that do NOT need fairchem/torch (importable in the fast tier):
model name-vs-path routing, task validation, and the dispersion double-count guard.
The full construction path (loading uma-s-1p2, dispersion wiring) is validated on the GPU box.
"""
import pytest

from interfaces.UMA_interface import (
    FUNCTIONAL,
    _adds_external_dispersion,
    _is_local_checkpoint,
    _validate_task,
)


@pytest.mark.parametrize("model,is_local", [
    ("uma-s-1p2", False),          # pretrained name -> get_predict_unit (downloaded by fairchem)
    ("uma-m-1p1", False),
    ("/data/ckpt/uma.pt", True),   # absolute path
    ("ckpts/uma.ckpt", True),      # has a path separator
    ("model.pth", True),           # checkpoint extension
])
def test_is_local_checkpoint(model, is_local):
    assert _is_local_checkpoint(model) is is_local


def test_validate_task_accepts_known_and_rejects_unknown():
    for t in FUNCTIONAL:                       # omol, omat, oc20, oc25, odac, omc
        _validate_task(t)                      # must not raise
    # the previous invalid default "default" (and any unknown task) must fail clearly,
    # not later as a KeyError on FUNCTIONAL[task]
    with pytest.raises(ValueError, match="unknown UMA task"):
        _validate_task("default")


@pytest.mark.parametrize("task,use_dispersion,expected", [
    ("omat", True, True),    # PBE materials head, no native dispersion -> add D3
    ("oc20", True, True),    # RPBE, no native dispersion -> add D3
    ("omol", True, False),   # wB97M-V / VV10 native -> must NOT double-count
    ("oc25", True, False),   # dispersion-corrected -> skip
    ("odac", True, False),
    ("omc", True, False),
    ("omat", False, False),  # not requested -> never add
    ("omol", False, False),
])
def test_adds_external_dispersion(task, use_dispersion, expected):
    assert _adds_external_dispersion(task, use_dispersion) is expected
