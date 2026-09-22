import torch

from sglang.srt.configs.model_config import minimax_sparse_layer_skips_topk
from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
    FlashMLAGfx938UnavailableError,
    build_flash_mla_sparse_prefill_k_end,
)


def _reference_k_end(q_lens, prefix_lens):
    values = []
    for q_len, prefix_len in zip(q_lens, prefix_lens):
        values.extend(prefix_len + offset + 1 for offset in range(q_len))
    return torch.tensor(values, dtype=torch.int32)


def test_build_flash_mla_sparse_prefill_k_end():
    q_lens = torch.tensor([0, 4, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 0, 4, 5], dtype=torch.int32)
    prefix_lens = torch.tensor([7, 11, 15], dtype=torch.int32)

    actual = build_flash_mla_sparse_prefill_k_end(
        cu_seqlens,
        prefix_lens,
        q_lens,
        5,
        torch.device("cpu"),
    )

    torch.testing.assert_close(actual, _reference_k_end([0, 4, 1], [7, 11, 15]))


def test_build_flash_mla_sparse_prefill_k_end_rejects_row_mismatch():
    q_lens = torch.tensor([2, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    prefix_lens = torch.tensor([10, 20], dtype=torch.int32)

    try:
        build_flash_mla_sparse_prefill_k_end(
            cu_seqlens,
            prefix_lens,
            q_lens,
            4,
            torch.device("cpu"),
        )
    except FlashMLAGfx938UnavailableError as err:
        assert "query rows" in str(err)
    else:
        raise AssertionError("expected mismatched query rows to be rejected")


def test_minimax_topk_frequency_is_anchored_at_first_sparse_layer():
    sparse_cfg = {"sparse_attention_freq": [0, 0, 0, 1, 1, 1, 1, 1, 1]}

    assert not minimax_sparse_layer_skips_topk(sparse_cfg, 3, 2)
    assert minimax_sparse_layer_skips_topk(sparse_cfg, 4, 2)
    assert not minimax_sparse_layer_skips_topk(sparse_cfg, 5, 2)
    assert not minimax_sparse_layer_skips_topk(sparse_cfg, 3, 4)
    assert minimax_sparse_layer_skips_topk(sparse_cfg, 4, 4)
    assert minimax_sparse_layer_skips_topk(sparse_cfg, 5, 4)
    assert minimax_sparse_layer_skips_topk(sparse_cfg, 6, 4)
    assert not minimax_sparse_layer_skips_topk(sparse_cfg, 7, 4)
    assert not minimax_sparse_layer_skips_topk(sparse_cfg, 2, 4)
    assert not minimax_sparse_layer_skips_topk(sparse_cfg, 4, 1)
