from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import quick_all_reduce


def _probe(monkeypatch, arch: str, opt_in: str | None = None) -> bool:
    quick_all_reduce.qr_rocm_arch_available.cache_clear()
    monkeypatch.setattr(quick_all_reduce, "_is_hip", True)
    if opt_in is None:
        monkeypatch.delenv("SGLANG_ALLOW_GFX938_QUICK_ALLREDUCE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ALLOW_GFX938_QUICK_ALLREDUCE", opt_in)
    with patch.object(
        quick_all_reduce.torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(gcnArchName=arch),
    ):
        return quick_all_reduce.qr_rocm_arch_available()


def test_gfx938_quick_reduce_requires_explicit_opt_in(monkeypatch):
    assert not _probe(monkeypatch, "gfx938:sramecc+:xnack-")
    assert _probe(monkeypatch, "gfx938:sramecc+:xnack-", "1")


def test_existing_rocm_quick_reduce_arches_remain_enabled(monkeypatch):
    assert _probe(monkeypatch, "gfx942:sramecc+:xnack-")
    assert _probe(monkeypatch, "gfx950:sramecc+:xnack-")
