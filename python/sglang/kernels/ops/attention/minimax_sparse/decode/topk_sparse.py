# Copyright 2025 XunhaoLai. All rights reserved.

import os
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from ..common.utils import (
    check_sparse_kv_fp8,
    robust_allocator,
    sparse_out_dtype,
    unit_scale,
)


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
        "BATCH_SIZE_BUCKET": lambda args: triton.next_power_of_2(args["batch_size"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [2, 3, 4, 5]
    ],
    key=["BATCH_SIZE_BUCKET", "gqa_group_size", "head_dim", "block_size", "HAS_SINK"],
)
@triton.jit
def _gqa_share_sparse_decode_kernel(
    q_ptr,  # Q: b x qh x d
    sink_ptr,  # Sink: qh x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    idx_ptr,  # topk index: qh x b x topk
    o_ptr,  # O partial: c x b x qh x d
    lse_ptr,  # lse partial: c x b x qh
    seq_lens,
    slot_ids,
    # shape
    max_slots,
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # Scale softmax probabilities into the useful e4m3 range before FP8 PV.
    p_scale,
    # stride
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # META parameters
    BATCH_SIZE_BUCKET: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    IS_FP8: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
):
    # decode program ids: split-K over the topk dimension to give every SM
    # something to do at small batch. pid(0) folds (batch, chunk) together so
    # the grid size = batch_size * NUM_TOPK_CHUNKS.
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    # per-chunk topk range. chunk_size is *runtime* (depends on max_topk which
    # is a runtime arg, not constexpr), so don't annotate as tl.constexpr —
    # doing so produces undefined behavior in Triton.
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk_compiletime = chunk_start_topk + chunk_size_topk
    # get q k start and len after rmpad
    seq_len = tl.minimum(tl.load(seq_lens + pid_b), max_kv_len)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # to avoid bugs when slot_ids is negative
    # get real topk
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk_idx = tl.load(idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1)
    valid_idx = tl.where(topk_idx >= 0, off_t, -1)
    real_topk = tl.sum(valid_idx != -1, axis=0)
    chunk_end_topk = tl.minimum(chunk_end_topk_compiletime, real_topk)
    # init pointer
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    # init statistics — kept at -inf so empty chunks (chunk_start >= real_topk)
    # naturally fall out as weight=0 in the merge step.
    if HAS_SINK and pid_c == 0:
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_q_h, stride_q_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sink_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_sink_h, stride_sink_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale  # (BLOCK_SIZE_H,)
        m_i = qsink
        lse_i = qsink
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_q_h, stride_q_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # only iterate over this chunk's topk slice. the load must respect the
    # per-chunk start offset.
    cur_idx_ptr = idx_base + chunk_start_topk * stride_ti_t
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        # load index
        c = tl.load(cur_idx_ptr).to(tl.int32) * BLOCK_SIZE_N
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        # resolve slots for this block via req_to_token
        pos = c + off_n
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        if DCP_SIZE > 1:
            pos_mask = pos_mask & (pos % DCP_SIZE == DCP_RANK)
            slots = slots // DCP_SIZE
        slots = (slots + max_slots) % max_slots  # safety against negative
        # load K as (head_dim, BLOCK_SIZE_N) via indirect addressing
        k_off = (
            slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_off,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            # fp8 KV cache: with bf16/fp16 Q this widens K to the compute dtype
            # (unit-scaled cache -> exact inverse dequant; k_scale covers
            # calibrated caches). With fp8 Q (fp8 attn-GEMM mode) the cast is a
            # no-op and tl.dot below runs fp8x8 on tensor cores. Matches the
            # bf16 path bit-for-bit when the cache is bf16 (IS_FP8 False ->
            # this branch is compiled out).
            k = k.to(q.dtype)
        # load V as (BLOCK_SIZE_N, head_dim) via indirect addressing
        v_off = (
            slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_off,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            # Cast V to the compute dtype. With bf16/fp16 Q this widens (so the
            # `p.to(v.dtype)` below keeps P in the compute dtype); with fp8 Q it
            # is a no-op and P is quantized to e4m3 for the fp8 PV MMA — the
            # same accuracy contract as fmha_sm100's fp8 kernel.
            v = v.to(q.dtype)
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        if DCP_SIZE > 1:
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        else:
            qk += tl.where(off_n[None, :] < seq_len - c, 0, float("-inf"))
        # [H, D], [D, N] -> [H, N]
        qk += tl.dot(q, k) * (sm_scale * k_scale)
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        if DCP_SIZE > 1:
            has_value = m_ij > float("-inf")
            safe_m_ij = tl.where(has_value, m_ij, 0.0)
            p = tl.where(
                qk > float("-inf"), tl.exp(qk - safe_m_ij[:, None]), 0.0
            )
            acc_o_scale = tl.where(
                m_i > float("-inf"), tl.exp(m_i - safe_m_ij), 0.0
            )
        else:
            p = tl.exp(qk - m_ij[:, None])
            acc_o_scale = tl.exp(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        # [H, N], [N, D] -> [H, D]
        acc_o += tl.dot((p * p_scale).to(v.dtype), v) * (v_scale / p_scale)
        # update statistics
        if DCP_SIZE > 1:
            lse_sum = tl.where(
                lse_i > float("-inf"), tl.exp(lse_i - safe_m_ij), 0.0
            ) + l_ij
            m_i = tl.where(has_value, m_ij, float("-inf"))
            lse_i = tl.where(
                has_value, safe_m_ij + tl.log(lse_sum), float("-inf")
            )
        else:
            m_i = m_ij
            lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
    # final scale (matches the old non-split kernel for chunks where lse_i>-inf).
    # For empty chunks (chunk_start_topk >= real_topk) the inner loop never
    # runs, so m_i = lse_i = -inf and naive `tl.exp(m_i - lse_i)` would compute
    # exp(-inf - (-inf)) = exp(NaN) = NaN, then 0 * NaN = NaN poisons o_partial
    # and the merge result. Gate the scale with tl.where so empty chunks emit a
    # clean zero (lse_i stays -inf which the merge correctly turns into weight=0).
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]
    # save partial output and lse for the merge step
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH, D] — partials in, merged out at chunk 0
    lse_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    # standard flash-decoding merge in linear (not log2) space, matching the
    # decode kernel which uses tl.exp / tl.log.
    lse_max = tl.max(lse, axis=0)
    has_value = lse_max > float("-inf")
    safe_lse_max = tl.where(has_value, lse_max, 0.0)
    weights = tl.where(has_value, tl.exp(lse - safe_lse_max), 0.0)
    weight_sum = tl.sum(weights, axis=0)
    weights = tl.where(has_value, weights / weight_sum, 0.0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    o_out_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_out_ptrs, o_merged.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)
    # Preserve the merged natural-log LSE in chunk zero for optional DCP
    # cross-rank output merging. Existing callers that only consume O are
    # unchanged.
    tl.store(
        lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h,
        tl.where(has_value, safe_lse_max + tl.log(weight_sum), float("-inf")),
    )


@triton.jit
def _gqa_share_sparse_multi_q_fused_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    req_to_token_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    seq_lens_ptr,
    slot_ids_ptr,
    max_slots,
    request_batch_size,
    gqa_group_size,
    head_dim,
    max_kv_len,
    sm_scale,
    k_scale,
    v_scale,
    p_scale,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    TOPK: tl.constexpr,
    VERIFY_GROUP_SIZE: tl.constexpr,
    FLAT_TOPK_SIZE: tl.constexpr,
    UNION_CHUNK_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Exact sparse Stage3 for a request's speculative query group.

    The old grouped-main experiment materialized a max-context-sized
    membership bitmap and launched separate scatter/compact kernels.  That was
    profitable only at very large local batch.  This kernel instead sorts the
    at-most 64 Top-K ids in registers, detects duplicates in place, and uses a
    per-query membership predicate while the selected K/V tile is resident.
    """

    pid_rc = tl.program_id(0)
    pid_r = pid_rc % request_batch_size
    pid_c = pid_rc // request_batch_size
    pid_kh = tl.program_id(1)

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_t = tl.arange(0, FLAT_TOPK_SIZE)

    verify_idx = off_h // gqa_group_size
    head_in_group = off_h % gqa_group_size
    valid_row = verify_idx < VERIFY_GROUP_SIZE
    flat_b = pid_r * VERIFY_GROUP_SIZE + verify_idx
    q_head = pid_kh * gqa_group_size + head_in_group

    row_seq_lens = tl.minimum(
        tl.load(seq_lens_ptr + flat_b, mask=valid_row, other=0).to(tl.int32),
        max_kv_len,
    )
    q = tl.load(
        q_ptr
        + flat_b[:, None] * stride_q_b
        + q_head[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=valid_row[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )

    topk_row = off_t // TOPK
    topk_col = off_t % TOPK
    valid_topk_lane = topk_row < VERIFY_GROUP_SIZE
    flat_topk = tl.load(
        idx_ptr
        + pid_kh * stride_ti_h
        + (pid_r * VERIFY_GROUP_SIZE + topk_row) * stride_ti_b
        + topk_col * stride_ti_t,
        mask=valid_topk_lane,
        other=-1,
    ).to(tl.int32)
    # -1 sentinels sort before valid non-negative block ids.  Valid candidates
    # therefore remain globally ascending, preserving each Q1 kernel's block
    # accumulation order after duplicates are removed.
    sorted_topk = tl.sort(flat_topk, dim=0, descending=False)

    dim_mask = off_d < head_dim
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    sid = (
        tl.load(slot_ids_ptr + pid_r * VERIFY_GROUP_SIZE).to(tl.int64)
        + max_slots
    ) % max_slots
    r2t_base = req_to_token_ptr + sid * stride_r2t_b

    chunk_start = pid_c * UNION_CHUNK_SIZE
    previous_from_sort = tl.sum(
        tl.where(off_t == chunk_start - 1, sorted_topk, 0), axis=0
    ).to(tl.int32)
    previous = tl.where(chunk_start > 0, previous_from_sort, -2)
    for chunk_offset in tl.static_range(0, UNION_CHUNK_SIZE):
        union_pos = chunk_start + chunk_offset
        candidate = tl.sum(
            tl.where(off_t == union_pos, sorted_topk, 0), axis=0
        ).to(tl.int32)
        unique_candidate = (candidate >= 0) & (candidate != previous)
        previous = candidate

        # Membership is exact even when Top-K order differs between queries.
        # The comparison is register-only; no max-context bitmap/workspace is
        # constructed and no metadata kernel is launched.
        member = tl.sum(
            (
                (flat_topk[None, :] == candidate)
                & valid_topk_lane[None, :]
                & (topk_row[None, :] == verify_idx[:, None])
            ).to(tl.int32),
            axis=1,
        ) > 0

        safe_candidate = tl.where(unique_candidate, candidate, 0)
        positions = safe_candidate * BLOCK_SIZE_N + off_n
        pos_mask = unique_candidate & (positions < max_kv_len)
        slots = tl.load(r2t_base + positions, mask=pos_mask, other=0).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr
            + slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        causal = (
            valid_row[:, None]
            & member[:, None]
            & pos_mask[None, :]
            & (positions[None, :] < row_seq_lens[:, None])
        )
        qk = tl.dot(q, k) * (sm_scale * k_scale)
        qk = tl.where(causal, qk, float("-inf"))
        block_m = tl.max(qk, axis=1)
        m_ij = tl.maximum(m_i, block_m)
        has_current = block_m > float("-inf")
        has_any = m_ij > float("-inf")
        safe_m = tl.where(has_any, m_ij, 0.0)
        p = tl.where(qk > float("-inf"), tl.exp(qk - safe_m[:, None]), 0.0)
        acc_scale = tl.where(m_i > float("-inf"), tl.exp(m_i - safe_m), 0.0)
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * acc_scale[:, None]
        acc_o += tl.dot((p * p_scale).to(v.dtype), v) * (v_scale / p_scale)
        lse_sum = (
            tl.where(lse_i > float("-inf"), tl.exp(lse_i - safe_m), 0.0)
            + l_ij
        )
        m_i = tl.where(has_current, m_ij, m_i)
        lse_i = tl.where(has_current, safe_m + tl.log(lse_sum), lse_i)

    has_value = lse_i > float("-inf")
    acc_o *= tl.where(has_value, tl.exp(m_i - lse_i), 0.0)[:, None]
    tl.store(
        o_ptr
        + pid_c * stride_o_c
        + flat_b[:, None] * stride_o_b
        + q_head[:, None] * stride_o_h
        + off_d[None, :] * stride_o_d,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=valid_row[:, None] & (off_d[None, :] < head_dim),
    )
    tl.store(
        lse_ptr
        + pid_c * stride_l_c
        + flat_b * stride_l_b
        + q_head * stride_l_h,
        lse_i,
        mask=valid_row,
    )


@torch.no_grad()
def _flash_decode_multi_q_fused(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_ids: torch.Tensor,
    block_size: int,
    topk_idx: torch.Tensor,
    sm_scale: float,
    k_scale: float,
    v_scale: float,
    p_scale: float,
    query_tile_size: int,
    return_lse: bool,
    is_fp8: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    flat_batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    request_batch = flat_batch // query_tile_size
    gqa_group_size = num_q_heads // num_kv_heads
    flat_topk_size = triton.next_power_of_2(query_tile_size * topk_idx.shape[2])
    block_size_h = max(
        16, triton.next_power_of_2(gqa_group_size * query_tile_size)
    )
    target_grid = 256
    target_chunks = max(
        1,
        min(
            topk_idx.shape[2],
            # Match the Q1 producer's split count. Each fused CTA owns two Q
            # rows, so matching its CTA count would duplicate sort/partial
            # work and over-split the union at the winner's local batch.
            target_grid // max(1, flat_batch * num_kv_heads),
        ),
    )
    num_union_chunks = 1 << (target_chunks.bit_length() - 1)
    union_chunk_size = flat_topk_size // num_union_chunks
    out_partial = torch.empty(
        (num_union_chunks, flat_batch, num_q_heads, v_cache.shape[-1]),
        dtype=sparse_out_dtype(q),
        device=q.device,
    )
    lse_partial = torch.empty(
        (num_union_chunks, flat_batch, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )
    _gqa_share_sparse_multi_q_fused_kernel[(
        request_batch * num_union_chunks,
        num_kv_heads,
    )](
        q,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        out_partial,
        lse_partial,
        seq_lens,
        slot_ids,
        k_cache.shape[0],
        request_batch,
        gqa_group_size,
        head_dim,
        req_to_token.shape[1],
        sm_scale,
        k_scale,
        v_scale,
        p_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        out_partial.stride(0),
        out_partial.stride(1),
        out_partial.stride(2),
        out_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        TOPK=topk_idx.shape[2],
        VERIFY_GROUP_SIZE=query_tile_size,
        FLAT_TOPK_SIZE=flat_topk_size,
        UNION_CHUNK_SIZE=union_chunk_size,
        IS_FP8=is_fp8,
        num_warps=8,
        num_stages=1,
    )
    _merge_topk_attn_out_kernel[(flat_batch, num_q_heads)](
        out_partial,
        lse_partial,
        head_dim,
        out_partial.stride(0),
        out_partial.stride(1),
        out_partial.stride(2),
        out_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        NUM_TOPK_CHUNKS=num_union_chunks,
    )
    out = out_partial[0].contiguous()
    if return_lse:
        return out, lse_partial[0].contiguous()
    return out


@torch.no_grad()
def flash_decode_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    return_lse: bool = False,
    verify_group_size: int = 1,
    use_multi_q_main: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="decode")
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    p_scale = (
        float(os.environ.get("SGLANG_M3_TRITON_FP8_P_SCALE", "448"))
        if q.dtype == torch.float8_e4m3fn
        else 1.0
    )
    if q.dtype == torch.float8_e4m3fn and not (0.0 < p_scale <= 448.0):
        raise ValueError(
            "SGLANG_M3_TRITON_FP8_P_SCALE must be in (0, 448] for e4m3fn, "
            f"got {p_scale}"
        )
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    assert slot_ids.shape[0] == batch_size and seq_lens.shape[0] == batch_size
    assert topk_idx.shape[0] == num_kv_heads
    assert (
        triton.next_power_of_2(block_size) == block_size
    ), f"block_size must be a power of 2, but got {block_size}"
    # assert slot_ids.max() < max_slots, f"get slot_ids {slot_ids}, but kv_cache shape is {kv_cache.shape}"
    max_kv_len = req_to_token.shape[1]
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    max_topk = topk_idx.shape[2]
    # sm scale
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    if (
        use_multi_q_main
        and verify_group_size in (2, 4)
        and batch_size % verify_group_size == 0
        and dcp_size == 1
        and sink is None
        and max_topk <= 16
        and head_dim == 128
        and v_cache.shape[-1] == 128
        and block_size == 128
        and gqa_group_size * 2 <= 64
    ):
        # Q4 is intentionally two adjacent Q2 tiles.  A single Q4/GQA16 CTA
        # creates a 64-row accumulator and made LLVM spend minutes optimizing
        # a multi-GiB compilation unit on gfx938.  Q2 preserves the most local
        # speculative-position KV reuse while keeping the production kernel
        # at 32 rows; the two pairs remain inside their original request.
        return _flash_decode_multi_q_fused(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            slot_ids=slot_ids,
            block_size=block_size,
            topk_idx=topk_idx,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            p_scale=p_scale,
            query_tile_size=2,
            return_lse=return_lse,
            is_fp8=is_fp8,
        )
    # Pick NUM_TOPK_CHUNKS so total grid ≈ TARGET_GRID. Same constraints as
    # flash_decode_with_topk_idx: must be power of 2 (Triton arange) and must
    # only depend on shape constants (so grid is fixed within a cuda graph).
    # Capped by max_topk because chunks beyond real_topk early-fall-through to
    # the merge-as-zero path; capping avoids wasting blocks at tiny topk.
    TARGET_GRID = 256
    target = max(
        1,
        min(max_topk, TARGET_GRID // max(1, batch_size * num_kv_heads)),
    )
    if verify_group_size > 1:
        override = os.environ.get("SGLANG_MINIMAX_MTP_NUM_TOPK_CHUNKS")
        if override is not None:
            try:
                requested = int(override)
            except ValueError:
                requested = 0
            if requested > 0:
                target = min(max_topk, requested)
    NUM_TOPK_CHUNKS = 1 << (target.bit_length() - 1)
    # output tensor: split-K partials, merged into chunk 0 by the merge kernel
    o_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        head_dim,
        dtype=sparse_out_dtype(q),
        device=q.device,
    )
    lse_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        dtype=torch.float32,
        device=q.device,
    )
    # launch attention kernel
    grid = (batch_size * NUM_TOPK_CHUNKS, num_kv_heads)
    _gqa_share_sparse_decode_kernel[grid](
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        slot_ids,
        max_slots,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        sm_scale,
        k_scale,
        v_scale,
        p_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size,
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
        IS_FP8=is_fp8,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
    )
    if NUM_TOPK_CHUNKS == 1:
        output = o_partial[0]
        if return_lse:
            return output, lse_partial[0]
        return output
    # merge partials into chunk 0
    merge_grid = (batch_size, num_q_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
    )
    output = o_partial[0].contiguous()
    if return_lse:
        return output, lse_partial[0].contiguous()
    return output
