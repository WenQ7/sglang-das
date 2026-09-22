import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    _compute_verify_topk_overlap_metrics,
)


def test_verify_topk_overlap_identical_sets():
    # [head, request-major (B * Q), topk]
    topk = torch.tensor(
        [[[1, 2, 3, 4], [1, 2, 3, 4], [10, 11, 12, 13], [10, 11, 12, 13]]],
        dtype=torch.int32,
    )
    metrics = _compute_verify_topk_overlap_metrics(topk, batch_size=2, verify_width=2)

    torch.testing.assert_close(metrics[:, 0], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(metrics[:, 1], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(metrics[:, 2], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(metrics[:, 3], torch.tensor([1.0, 1.0]))


def test_verify_topk_overlap_partial_and_disjoint_sets():
    topk = torch.tensor(
        [[[1, 2, 3, 4], [3, 4, 5, 6], [10, 11, -1, -1], [20, 21, -1, -1]]],
        dtype=torch.int32,
    )
    metrics = _compute_verify_topk_overlap_metrics(topk, batch_size=2, verify_width=2)

    torch.testing.assert_close(metrics[:, 0], torch.tensor([6.0, 4.0]))
    torch.testing.assert_close(metrics[:, 1], torch.tensor([1.5, 2.0]))
    torch.testing.assert_close(metrics[:, 2], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(metrics[:, 3], torch.tensor([1.0 / 3.0, 0.0]))


def test_verify_topk_overlap_rejects_non_request_major_shape():
    topk = torch.zeros((1, 5, 4), dtype=torch.int32)
    try:
        _compute_verify_topk_overlap_metrics(topk, batch_size=2, verify_width=2)
    except ValueError as err:
        assert "request-major" in str(err)
    else:
        raise AssertionError("expected invalid flattened verify rows to be rejected")
