"""FP8 draft LM-head followed by memory-local deterministic Top-1.

The gfx938 Triton compiler in the current DAS stack cannot lower one monolithic
OCP FP8 ``tl.dot`` over the complete vocabulary with a reduction epilogue. The
fused implementation therefore partitions vocabulary N, reduces every
register-resident GEMM tile, then communicates only one candidate per TP rank.
Both implementations eliminate the global ``[tokens, 200064]`` logits tensor
and vocabulary AllGather. ``fp8_lm_head_top1`` materializes one local TP shard;
``fp8_lm_head_top1_fused`` reduces register-resident GEMM tiles and never writes
that local logits shard.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.topk1 import draft_topk1_argmax


_FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _fp8_lm_head_top1_partials_kernel(
    x_ptr,
    weight_ptr,
    x_scale_ptr,
    weight_scale_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    num_n_blocks: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one vocabulary-tile winner without writing tile logits.

    The kernel deliberately keeps the GEMM accumulator in registers and applies
    dynamic-token/channel-weight scales before reducing N.  Its only global
    output is one ``(value, vocab_id)`` candidate per row and N tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        current_k = k_start + offs_k
        mask_k = current_k < k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + offs_n[None, :] * stride_wn
            + current_k[:, None] * stride_wk,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(x, weight, acc=acc)

    if APPLY_SCALE:
        x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=0.0)
        weight_scale = tl.load(
            weight_scale_ptr + offs_n, mask=mask_n, other=0.0
        )
        acc *= x_scale[:, None] * weight_scale[None, :]
    # Both the ordinary BF16 LM-head and LightOp's channel-FP8 LM-head expose
    # BF16 logits. Match that rounding before Top-1.
    logits = acc.to(tl.bfloat16)
    logits = logits.to(tl.float32)
    logits = tl.where(mask_n[None, :] & (logits == logits), logits, -float("inf"))
    local_index = tl.argmax(logits, axis=1, tie_break_left=True)
    local_value = tl.max(logits, axis=1)
    output_offset = offs_m * num_n_blocks + pid_n
    tl.store(partial_values_ptr + output_offset, local_value, mask=mask_m)
    tl.store(
        partial_indices_ptr + output_offset,
        pid_n * BLOCK_N + local_index,
        mask=mask_m,
    )


@triton.jit
def _fp8_lm_head_top1_finalize_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_values_ptr,
    output_indices_ptr,
    num_n_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_n_blocks
    values = tl.load(
        partial_values_ptr + row * num_n_blocks + offsets,
        mask=mask,
        other=-float("inf"),
    )
    winner = tl.argmax(values, axis=0, tie_break_left=True)
    tl.store(output_values_ptr + row, tl.max(values, axis=0))
    tl.store(
        output_indices_ptr + row,
        tl.load(partial_indices_ptr + row * num_n_blocks + winner),
    )


def quantize_lm_head_weight_fp8_per_channel(
    weight: torch.Tensor, *, chunk_rows: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize row-major ``[local_vocab, hidden]`` weights once after load."""
    if weight.ndim != 2 or not weight.is_cuda:
        raise ValueError("draft FP8 LM head requires a 2-D device weight")
    qweight = torch.empty_like(weight, dtype=_FP8_DTYPE)
    scales = torch.empty(
        (weight.shape[0], 1), dtype=torch.float32, device=weight.device
    )
    fp8_max = torch.finfo(_FP8_DTYPE).max
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        chunk = weight[start:end]
        scale = chunk.abs().float().amax(dim=1).clamp_min_(1e-12) / fp8_max
        qweight[start:end].copy_((chunk.float() / scale[:, None]).to(_FP8_DTYPE))
        scales[start:end, 0].copy_(scale)
    return qweight, scales


def fp8_lm_head_top1(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    valid_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local maximum values/ids without creating global vocab logits."""
    from sglang.srt.layers.quantization.fp8_utils import (
        apply_fp8_lightop_channelwise_linear,
    )

    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden_states and weight must both be 2-D")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError(
            f"LM-head K mismatch: {hidden_states.shape[1]} vs {weight.shape[1]}"
        )
    if weight.dtype != _FP8_DTYPE:
        raise ValueError(f"expected {_FP8_DTYPE} LM-head weight, got {weight.dtype}")
    if not 0 < valid_vocab_size <= weight.shape[0]:
        raise ValueError(
            f"invalid local vocab {valid_vocab_size} for {weight.shape[0]} rows"
        )
    local_logits = apply_fp8_lightop_channelwise_linear(
        hidden_states.contiguous(), weight, weight_scale
    )
    # Padding rows are zero-filled by the vocab loader and can incorrectly win
    # if every real logit is negative, so exclude them from the reduction.
    return draft_topk1_argmax(local_logits[:, :valid_vocab_size].contiguous())


def fp8_lm_head_top1_fused(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    valid_vocab_size: int,
    block_n: int = 128,
    block_k: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 LM-head GEMM + Top-1 without materializing local logits.

    This is an opt-in gfx938-oriented implementation for EAGLE3's very small
    draft batches.  It emits ``ceil(local_vocab / block_n)`` candidates per
    row, rather than ``local_vocab`` BF16 logits.  The final reduction preserves
    ``torch.argmax``'s lowest-token-id tie breaking because candidates are
    ordered by monotonically increasing vocabulary tile.
    """
    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden_states and weight must both be 2-D")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError(
            f"LM-head K mismatch: {hidden_states.shape[1]} vs {weight.shape[1]}"
        )
    if weight.dtype != _FP8_DTYPE:
        raise ValueError(f"expected {_FP8_DTYPE} LM-head weight, got {weight.dtype}")
    if weight_scale.numel() != weight.shape[0]:
        raise ValueError(
            "weight_scale must contain one scale per LM-head output channel"
        )
    if not 0 < valid_vocab_size <= weight.shape[0]:
        raise ValueError(
            f"invalid local vocab {valid_vocab_size} for {weight.shape[0]} rows"
        )
    if block_n not in (64, 128, 256) or block_k not in (64, 128, 256):
        raise ValueError("block_n and block_k must be one of 64, 128, or 256")

    # Reuse the same quantizer as the tuned LightOp channel-FP8 GEMM so the
    # fused and materialized paths compare with identical input numerics.
    from lightop.quant.fp8 import per_token_quant_fp8

    hidden_states = hidden_states.contiguous()
    qinput, x_scale = per_token_quant_fp8(hidden_states, dtype=_FP8_DTYPE)
    m, k = qinput.shape
    if m == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=hidden_states.device),
            torch.empty((0,), dtype=torch.int32, device=hidden_states.device),
        )
    x_scale = x_scale.reshape(-1).contiguous()
    weight_scale = weight_scale.reshape(-1).contiguous()
    num_n_blocks = triton.cdiv(valid_vocab_size, block_n)
    partial_values = torch.empty(
        (m, num_n_blocks), dtype=torch.float32, device=hidden_states.device
    )
    partial_indices = torch.empty(
        (m, num_n_blocks), dtype=torch.int32, device=hidden_states.device
    )
    output_values = torch.empty(
        (m,), dtype=torch.float32, device=hidden_states.device
    )
    output_indices = torch.empty(
        (m,), dtype=torch.int32, device=hidden_states.device
    )
    block_m = triton.next_power_of_2(m)
    if block_m > 8:
        raise ValueError(
            f"fused draft LM-head is intended for M <= 8, got M={m}"
        )

    _fp8_lm_head_top1_partials_kernel[(triton.cdiv(m, block_m), num_n_blocks)](
        qinput,
        weight,
        x_scale,
        weight_scale,
        partial_values,
        partial_indices,
        m,
        valid_vocab_size,
        k,
        qinput.stride(0),
        qinput.stride(1),
        weight.stride(0),
        weight.stride(1),
        num_n_blocks,
        APPLY_SCALE=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    _fp8_lm_head_top1_finalize_kernel[(m,)](
        partial_values,
        partial_indices,
        output_values,
        output_indices,
        num_n_blocks,
        BLOCK=triton.next_power_of_2(num_n_blocks),
        num_warps=4,
    )
    return output_values, output_indices


def bf16_lm_head_top1_fused(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    valid_vocab_size: int,
    block_n: int = 128,
    block_k: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 LM-head GEMM + Top-1 without materializing local logits."""
    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden_states and weight must both be 2-D")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 fused draft LM-head requires BF16 input and weight")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError(
            f"LM-head K mismatch: {hidden_states.shape[1]} vs {weight.shape[1]}"
        )
    if not 0 < valid_vocab_size <= weight.shape[0]:
        raise ValueError(
            f"invalid local vocab {valid_vocab_size} for {weight.shape[0]} rows"
        )
    hidden_states = hidden_states.contiguous()
    weight = weight.contiguous()
    m, k = hidden_states.shape
    if m == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=hidden_states.device),
            torch.empty((0,), dtype=torch.int32, device=hidden_states.device),
        )
    block_m = triton.next_power_of_2(m)
    if block_m > 8:
        raise ValueError(f"fused draft LM-head is intended for M <= 8, got M={m}")
    num_n_blocks = triton.cdiv(valid_vocab_size, block_n)
    partial_values = torch.empty(
        (m, num_n_blocks), dtype=torch.float32, device=hidden_states.device
    )
    partial_indices = torch.empty(
        (m, num_n_blocks), dtype=torch.int32, device=hidden_states.device
    )
    output_values = torch.empty(
        (m,), dtype=torch.float32, device=hidden_states.device
    )
    output_indices = torch.empty(
        (m,), dtype=torch.int32, device=hidden_states.device
    )
    _fp8_lm_head_top1_partials_kernel[(triton.cdiv(m, block_m), num_n_blocks)](
        hidden_states,
        weight,
        hidden_states,
        weight,
        partial_values,
        partial_indices,
        m,
        valid_vocab_size,
        k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        num_n_blocks,
        APPLY_SCALE=False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    _fp8_lm_head_top1_finalize_kernel[(m,)](
        partial_values,
        partial_indices,
        output_values,
        output_indices,
        num_n_blocks,
        BLOCK=triton.next_power_of_2(num_n_blocks),
        num_warps=4,
    )
    return output_values, output_indices
