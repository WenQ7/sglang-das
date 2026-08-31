import torch

import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def test_hip_per_token_quant_prefers_exported_aot_op(monkeypatch):
    called = {}

    def fake_aot(x, x_q, x_s):
        called["x"] = x
        x_q.zero_()
        x_s.fill_(1)

    monkeypatch.setattr(fp8_kernel, "_is_hip", True)
    monkeypatch.setattr(fp8_kernel, "_hip_per_token_quant_fp8", fake_aot)
    x = torch.ones((2, 8), dtype=torch.bfloat16)

    x_q, x_s = fp8_kernel.sglang_per_token_quant_fp8(x)

    assert called["x"] is x
    assert x_q.shape == x.shape
    assert x_s.shape == (2, 1)


def test_hip_per_token_quant_falls_back_only_when_aot_op_missing(monkeypatch):
    expected = (object(), object())
    captured = {}

    def fake_triton(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(fp8_kernel, "_is_hip", True)
    monkeypatch.setattr(fp8_kernel, "_hip_per_token_quant_fp8", None)
    monkeypatch.setattr(fp8_kernel, "_per_token_group_quant_8bit_raw", fake_triton)
    x = torch.ones((2, 7), dtype=torch.bfloat16)

    assert fp8_kernel.sglang_per_token_quant_fp8(x) is expected
    assert captured["x"] is x
    assert captured["group_size"] == 7
