"""CPU-only tests for target-verify direct-Top1 admission and fallback."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    is_target_verify_greedy_top1_eligible,
)
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _batch(**sampling_overrides):
    sampling = dict(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        logit_bias=None,
        grammar_mask=None,
        grammars=None,
        return_sampling_masks=None,
    )
    sampling.update(sampling_overrides)
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        return_logprob=False,
        sampling_info=SimpleNamespace(**sampling),
        replace_embeds=None,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"is_all_greedy": False},
        {"has_custom_logit_processor": True},
        {"acc_additive_penalties": object()},
        {"acc_scaling_penalties": object()},
        {"penalizer_orchestrator": SimpleNamespace(is_required=True)},
        {"logit_bias": object()},
        {"grammar_mask": object()},
        {"grammars": [None, object()]},
        {"return_sampling_masks": [False, True]},
    ],
)
def test_unsupported_sampling_modifier_falls_back(override):
    assert not is_target_verify_greedy_top1_eligible(_batch(**override))


def test_plain_greedy_batch_is_eligible():
    assert is_target_verify_greedy_top1_eligible(_batch())


def test_graph_capture_batch_without_sampling_metadata_is_eligible():
    batch = _batch()
    batch.sampling_info = None
    assert is_target_verify_greedy_top1_eligible(batch)


def test_direct_top1_graph_replay_falls_back_before_graph_lookup():
    runner = SimpleNamespace(target_verify_direct_top1=True)
    runner._can_run_ragged_verify_graph = MagicMock()
    assert not DecodeCudaGraphRunner.can_run_graph(
        runner, _batch(is_all_greedy=False)
    )
    runner._can_run_ragged_verify_graph.assert_not_called()


def test_dp_sync_payload_carries_top1_admission_before_adaptive_stats():
    info = MLPSyncBatchInfo(
        dp_size=8,
        tp_size=1,
        cp_size=1,
        num_tokens=4,
        num_tokens_for_logprob=4,
        cp_num_tokens=0,
        can_run_decode_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
        target_verify_greedy_top1_eligible=False,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=1,
        adaptive_accept_sum=7,
        adaptive_request_count=2,
        adaptive_max_local_batch=3,
        adaptive_sync_enabled=True,
    )
    payload = info._get_local_tensor(device="cpu")
    assert payload[8].item() == 0
    assert payload[9:].tolist() == [7, 2, 3]


@pytest.mark.parametrize(
    ("use_fp32_lm_head", "expected_dtype"),
    [(False, torch.bfloat16), (True, torch.float32)],
)
def test_empty_lm_head_short_circuits_zero_row_gemm(
    use_fp32_lm_head, expected_dtype
):
    processor = SimpleNamespace(use_fp32_lm_head=use_fp32_lm_head)
    lm_head = SimpleNamespace(weight=torch.empty(17, 8, dtype=torch.bfloat16))
    hidden_states = torch.empty(0, 8, dtype=torch.bfloat16)

    logits = LogitsProcessor._compute_lm_head(processor, hidden_states, lm_head)

    assert logits.shape == (0, 17)
    assert logits.dtype == expected_dtype


def test_empty_lm_head_short_circuits_batched_zero_row_gemm():
    processor = SimpleNamespace(use_fp32_lm_head=False)
    lm_head = SimpleNamespace(weight=torch.empty(17, 8, dtype=torch.bfloat16))
    hidden_states = torch.empty(8, 0, 8, dtype=torch.bfloat16)

    logits = LogitsProcessor._compute_lm_head(processor, hidden_states, lm_head)

    assert logits.shape == (8, 0, 17)
