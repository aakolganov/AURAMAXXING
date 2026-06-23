"""Tests for thread/oversubscription controls (runner.threads)."""
import os

import pytest

from runner.threads import set_thread_env, set_torch_threads, configure_threads, _THREAD_ENV


@pytest.fixture
def restore_thread_env():
    """Snapshot and restore the thread env vars so a test never leaks into others."""
    saved = {var: os.environ.get(var) for var in _THREAD_ENV}
    yield
    for var, val in saved.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def test_set_thread_env_sets_every_var(restore_thread_env):
    set_thread_env(3)
    assert all(os.environ[var] == "3" for var in _THREAD_ENV)


def test_configure_threads_none_is_noop(restore_thread_env):
    before = {var: os.environ.get(var) for var in _THREAD_ENV}
    configure_threads(None)
    assert {var: os.environ.get(var) for var in _THREAD_ENV} == before


def test_configure_threads_rejects_nonpositive(restore_thread_env):
    with pytest.raises(ValueError):
        configure_threads(0)


def test_set_torch_threads_takes_effect():
    torch = pytest.importorskip("torch")
    original = torch.get_num_threads()
    try:
        set_torch_threads(2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)
