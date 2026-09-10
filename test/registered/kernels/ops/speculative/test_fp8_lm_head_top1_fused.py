import unittest

import torch

from sglang.kernels.ops.speculative.fp8_lm_head_top1 import (
    bf16_lm_head_top1_fused,
    fp8_lm_head_top1_fused,
    quantize_lm_head_weight_fp8_per_channel,
)
from sglang.kernels.ops.speculative.topk1 import draft_topk1_argmax


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None,
    "requires a ROCm GPU",
)
class TestFp8LmHeadTop1Fused(unittest.TestCase):
    def setUp(self):
        try:
            from lightop.quant.fp8 import per_token_quant_fp8  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"LightOp FP8 quantizer is unavailable: {exc}")

    def test_matches_dequantized_reference_with_clear_margin(self):
        torch.manual_seed(20260910)
        m, n, k = 4, 513, 256
        hidden = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01
        # Give each row a deliberately dominant and distinct output channel.
        winners = torch.tensor([3, 129, 257, 511], device="cuda")
        for row, winner in enumerate(winners.tolist()):
            weight[winner].copy_(hidden[row] * 8)
        qweight, scales = quantize_lm_head_weight_fp8_per_channel(weight)

        _, actual = fp8_lm_head_top1_fused(
            hidden, qweight, scales, valid_vocab_size=n
        )

        self.assertTrue(torch.equal(actual.long(), winners))

    def test_bf16_matches_materialized_logits_exactly(self):
        torch.manual_seed(17)
        m, n, k = 8, 513, 256
        hidden = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01

        expected_values, expected_ids = draft_topk1_argmax(
            hidden @ weight.T
        )
        actual_values, actual_ids = bf16_lm_head_top1_fused(
            hidden, weight, valid_vocab_size=n
        )

        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.equal(actual_values, expected_values))

    def test_excludes_padding_rows(self):
        m, valid_n, padded_n, k = 2, 257, 384, 256
        hidden = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
        weight = -torch.ones(
            (padded_n, k), device="cuda", dtype=torch.bfloat16
        )
        # A padded zero row would beat every valid negative logit if it were
        # accidentally included in Top-1.
        weight[valid_n:].zero_()
        qweight, scales = quantize_lm_head_weight_fp8_per_channel(weight)

        _, actual = fp8_lm_head_top1_fused(
            hidden, qweight, scales, valid_vocab_size=valid_n
        )

        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_lowest_id_wins_ties(self):
        hidden = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((257, 256), device="cuda", dtype=torch.bfloat16)
        qweight, scales = quantize_lm_head_weight_fp8_per_channel(weight)

        _, actual = fp8_lm_head_top1_fused(
            hidden, qweight, scales, valid_vocab_size=257
        )

        self.assertEqual(actual.item(), 0)

    def test_rejects_non_draft_batch(self):
        hidden = torch.zeros((9, 256), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((128, 256), device="cuda", dtype=torch.bfloat16)
        qweight, scales = quantize_lm_head_weight_fp8_per_channel(weight)

        with self.assertRaisesRegex(ValueError, "M <= 8"):
            fp8_lm_head_top1_fused(
                hidden, qweight, scales, valid_vocab_size=128
            )

    def test_hip_graph_replay_uses_new_input(self):
        torch.manual_seed(11)
        m, n, k = 2, 513, 256
        graph_input = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01
        # Ensure the two replay inputs have stable, unambiguous, distinct ids.
        weight[7].copy_(graph_input[0] * 8)
        weight[131].copy_(graph_input[1] * 8)
        qweight, scales = quantize_lm_head_weight_fp8_per_channel(weight)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            fp8_lm_head_top1_fused(
                graph_input, qweight, scales, valid_vocab_size=n
            )
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, graph_ids = fp8_lm_head_top1_fused(
                graph_input, qweight, scales, valid_vocab_size=n
            )
        graph.replay()
        torch.cuda.synchronize()
        first = graph_ids.clone()

        graph_input.copy_(-graph_input)
        graph.replay()
        torch.cuda.synchronize()
        second = graph_ids.clone()

        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
