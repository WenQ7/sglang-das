from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention import hcu_flash_attention as hcu_fa
from sglang.srt.server_args import (
    ATTENTION_BACKEND_CHOICES,
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
)


def _paged_inputs():
    q = torch.empty((4, 8, 128), dtype=torch.bfloat16)
    k = torch.empty((8, 64, 1, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu_q = torch.tensor([0, 4], dtype=torch.int32)
    seq_lens = torch.tensor([512], dtype=torch.int32)
    page_table = torch.zeros((1, 8), dtype=torch.int32)
    return q, k, v, cu_q, seq_lens, page_table


def test_hcu_paged_adapter_maps_sglang_metadata_to_bshd_unified():
    q, k, v, cu_q, seq_lens, page_table = _paged_inputs()
    out = torch.empty_like(q)
    calls = []

    def fake_unified(**kwargs):
        calls.append(kwargs)
        return kwargs["out"]

    with patch.object(hcu_fa, "_hcu_flash_ops", return_value=(None, fake_unified)):
        result = hcu_fa.flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=seq_lens,
            page_table=page_table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=4,
            out=out,
        )

    assert result is out
    assert len(calls) == 1
    call = calls[0]
    assert call["layout"] == "bshd"
    assert call["max_seqlen_q"] == 4
    assert call["max_seqlen_k"] == 512
    assert call["seqused_k"].dtype == torch.int32
    assert call["block_table"].dtype == torch.int32


def test_hcu_paged_adapter_rejects_fused_cache_update():
    q, k, v, cu_q, seq_lens, page_table = _paged_inputs()
    with pytest.raises(NotImplementedError, match="does not support k="):
        hcu_fa.flash_attn_with_kvcache(
            q,
            k,
            v,
            k=torch.empty((4, 1, 128), dtype=torch.bfloat16),
            cache_seqlens=seq_lens,
            page_table=page_table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=4,
        )


def test_hcu_raw_varlen_copies_to_sglang_out_buffer():
    q = torch.empty((4, 8, 128), dtype=torch.bfloat16)
    k = torch.empty((4, 1, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    out = torch.empty_like(q)
    raw_result = torch.full_like(q, 3)

    def fake_raw(*args, **kwargs):
        return raw_result

    with patch.object(hcu_fa, "_hcu_flash_ops", return_value=(fake_raw, None)):
        result = hcu_fa.flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=4,
            max_seqlen_k=4,
            out=out,
        )

    assert result is out
    torch.testing.assert_close(out, raw_result)


def test_hcu_backend_is_reachable_from_serving_cli_and_chunked_prefix_cache():
    assert "hcu_fa" in ATTENTION_BACKEND_CHOICES
    assert "hcu_fa" in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
