import pytest

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    _aiter_enable_register_for_capturing,
)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, True), ("1", True), ("true", True), ("0", False), ("false", False)],
)
def test_aiter_register_capture_honors_environment(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool
) -> None:
    if env_value is None:
        monkeypatch.delenv("AITER_AR_ENABLE_REG_CAPTURE", raising=False)
    else:
        monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", env_value)
    assert _aiter_enable_register_for_capturing(False) is expected


def test_aiter_register_capture_disabled_by_memory_saver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", "1")
    assert not _aiter_enable_register_for_capturing(True)
