# Copyright 2025 XunhaoLai. All rights reserved.

import logging
import os
from typing import Callable, List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.index import topk_index_reduce
from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    build_query_group_topk_union,
    flash_prefill_with_gqa_share_sparse,
)
from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mha

logger = logging.getLogger(__name__)
_msa_fallback_warned = False


def _flatten_main_kv_cache(cache: torch.Tensor) -> torch.Tensor:
    """Normalize paged main KV cache to the slot-major layout sparse kernels use."""
    if cache.dim() == 3:
        return cache
    if cache.dim() != 4:
        raise ValueError(
            "MiniMax sparse attention expects a 3D slot cache or a 4D paged "
            f"cache, got shape={tuple(cache.shape)}"
        )
    if not cache.is_contiguous():
        raise ValueError(
            "MiniMax sparse attention requires a contiguous 4D paged cache for "
            f"slot flattening, got shape={tuple(cache.shape)}, stride={cache.stride()}"
        )
    # [num_pages, page_size, num_kv_heads, head_dim]
    return cache.view(-1, cache.shape[-2], cache.shape[-1])


def _warn_msa_fallback(err: Exception) -> None:
    global _msa_fallback_warned
    if _msa_fallback_warned:
        return
    logger.warning(
        "MiniMax MSA backend is unavailable (%s); falling back to Triton sparse attention.",
        err,
    )
    _msa_fallback_warned = True


def minimax_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    dcp_group=None,
    logical_max_seqlen_q: Optional[int] = None,
):
    """Run MiniMax-M3 sparse prefill.

    ``cu_seqblocks_q``, ``max_seqblock_q``, and ``all_seqblock_q`` are optional
    precomputed query-block metadata shared by the index and value sparse
    kernels. Supplying them avoids recomputing the same block layout twice.
    ``seqlens_cpu`` (host copy of ``torch.diff(cu_seqlens)``) is forwarded to
    ``get_cu_seqblocks`` to avoid a per-layer device sync when it recomputes.
    """
    k_cache = _flatten_main_kv_cache(k_cache)
    v_cache = _flatten_main_kv_cache(v_cache)
    if dcp_size > 1:
        if dcp_size != 2:
            raise NotImplementedError(
                "MiniMax-M3 DCP prefill currently supports DCP2 only"
            )
        if dcp_group is None or dcp_group.world_size != dcp_size:
            raise ValueError("MiniMax-M3 DCP2 prefill requires its matching group")
        if score_type != "max" or not disable_index_value:
            raise NotImplementedError(
                "MiniMax-M3 DCP2 prefill requires score_type='max' and "
                "disable_index_value=True"
            )
        if use_msa or sink is not None or idx_sink is not None:
            raise NotImplementedError(
                "MiniMax-M3 DCP2 prefill requires Triton sparse attention "
                "without sinks"
            )
    # The main sparse kernel tiles Q and all GQA heads together and requires
    # BLOCK_SIZE_Q * gqa_group_size <= 128.  TP/DP choices change the local GQA
    # group, so clamp the requested query block to the largest supported power
    # of two instead of making one launch-script value fail under another
    # parallel layout.
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"MiniMax sparse GQA mismatch: q_heads={num_q_heads}, "
            f"kv_heads={num_kv_heads}"
        )
    gqa_group_size = num_q_heads // num_kv_heads
    main_max_qh = int(os.environ.get("SGLANG_MINIMAX_PREFILL_MAIN_MAX_QH", "128"))
    if main_max_qh not in (128, 256):
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_MAIN_MAX_QH must be 128 or 256, "
            f"got {main_max_qh}"
        )
    max_block_size_q = max(1, main_max_qh // (gqa_group_size * dcp_size))
    while block_size_q > max_block_size_q:
        block_size_q //= 2
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    exact_grouped_main_q = int(
        os.environ.get("SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_Q", "1")
    )
    exact_grouped_main_query_len = (
        max_seqlen_q
        if logical_max_seqlen_q is None
        else int(logical_max_seqlen_q)
    )
    exact_grouped_main_min_query_len = int(
        os.environ.get(
            "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_MIN_QUERY_LEN", "4096"
        )
    )
    exact_grouped_main_max_query_len = int(
        os.environ.get(
            "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_MAX_QUERY_LEN", "32768"
        )
    )
    if exact_grouped_main_q < 1:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_Q must be >= 1, "
            f"got {exact_grouped_main_q}"
        )
    if exact_grouped_main_q & (exact_grouped_main_q - 1):
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_Q must be a power of two, "
            f"got {exact_grouped_main_q}"
        )
    if exact_grouped_main_min_query_len < 0:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_MIN_QUERY_LEN must be "
            f">= 0, got {exact_grouped_main_min_query_len}"
        )
    if exact_grouped_main_max_query_len < 0:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_MAX_QUERY_LEN must be "
            f">= 0, got {exact_grouped_main_max_query_len}"
        )
    while exact_grouped_main_q > max_block_size_q:
        exact_grouped_main_q //= 2
    use_exact_grouped_main = (
        block_size_q == 1
        and exact_grouped_main_q > 1
        and (
            exact_grouped_main_min_query_len == 0
            or exact_grouped_main_query_len >= exact_grouped_main_min_query_len
        )
        and (
            exact_grouped_main_max_query_len == 0
            or exact_grouped_main_query_len <= exact_grouped_main_max_query_len
        )
        and dcp_size == 1
    )
    fused_topk_union = (
        os.environ.get("SGLANG_MINIMAX_PREFILL_FUSED_TOPK_UNION", "0") == "1"
    )

    # All seqlen is less than topk, use full attention
    # Step 1: Flash attention with topk index (using index head)
    idx_o, topk_idx = flash_prefill_with_topk_index(
        q=idx_q,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        sink=idx_sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
        q_scale=idx_q_scale,
        k_scale=idx_k_scale,
        v_scale=idx_v_scale,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_group=dcp_group,
        # The fused union accepts score-ranked AITER ids and emits a sorted
        # unique block-id row itself.  Keep sorting for every other consumer.
        sort_topk_ids=not (use_exact_grouped_main and fused_topk_union),
    )
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )
    q_for_main = (
        dcp_group.all_gather(q.contiguous(), dim=1).contiguous()
        if dcp_size > 1
        else q
    )
    main_block_size_q = block_size_q
    main_cu_seqblocks_q = cu_seqblocks_q
    main_max_seqblock_q = max_seqblock_q
    per_query_topk_idx = None
    query_group_mask = None
    # Query-sharded prefill CP passes a local max_seqlen_q (16K / CP8 = 2K).
    # Selection thresholds describe the user-visible logical request, so use
    # the pre-sharding length supplied by the backend when it is available.
    # Exact Q1 index selection and grouped main attention are independent.
    # The indexer above still computes one score row and one exact Top-K set for
    # every query.  Grouping only unions those already-final sets for the main
    # attention kernel, whose per-query membership mask excludes every block
    # not selected by that query.  Therefore this changes scheduling and KV
    # reuse, not sparse-attention semantics.  Limit the optimization to the
    # long cache-hit extend window by default: using it while constructing the
    # 128K causal prefix changes a handful of BF16 reductions and then persists
    # those differences in the KV cache.  A zero length bound disables that
    # side of the range check.
    if use_exact_grouped_main and topk_idx.shape[1] == q.shape[0]:
        (
            main_cu_seqblocks_q,
            main_max_seqblock_q,
            main_all_seqblock_q,
            _,
            _,
            _,
        ) = get_cu_seqblocks(
            cu_seqlens,
            max_seqlen_q,
            exact_grouped_main_q,
            block_size_k,
            seqlens_cpu,
        )
        per_query_topk_idx = topk_idx
        precompute_union_bitmask = (
            os.environ.get("SGLANG_MINIMAX_PREFILL_UNION_BITMASK", "0") == "1"
        )
        grouped_union = build_query_group_topk_union(
            topk_idx,
            cu_seqlens,
            main_cu_seqblocks_q,
            exact_grouped_main_q,
            all_groups=main_all_seqblock_q,
            max_num_blocks=(max_seqlen_k + block_size_k - 1) // block_size_k,
            max_union=int(
                os.environ.get(
                    "SGLANG_MINIMAX_PREFILL_EXACT_GROUPED_MAIN_MAX",
                    str(exact_grouped_main_q * topk),
                )
            ),
            return_query_mask=precompute_union_bitmask,
        )
        if precompute_union_bitmask:
            topk_idx, query_group_mask = grouped_union
        else:
            topk_idx = grouped_union
        main_block_size_q = exact_grouped_main_q
    elif block_size_q > 1 and topk_idx.shape[1] == q.shape[0]:
        if os.environ.get("SGLANG_MINIMAX_PREFILL_GROUPED_UNION_MAIN", "0") == "1":
            per_query_topk_idx = topk_idx
            topk_idx = build_query_group_topk_union(
                topk_idx,
                cu_seqlens,
                cu_seqblocks_q,
                block_size_q,
                all_groups=all_seqblock_q,
                max_num_blocks=(max_seqlen_k + block_size_k - 1) // block_size_k,
                max_union=int(
                    os.environ.get(
                        "SGLANG_MINIMAX_PREFILL_GROUPED_UNION_MAX",
                        str(block_size_q * topk),
                    )
                ),
            )
        else:
            # Candidate reranking emits one Top-K row for every query.  Use the
            # established Q1 consumer when grouped-union attention is disabled.
            main_block_size_q = 1
            (
                main_cu_seqblocks_q,
                main_max_seqblock_q,
                _,
                _,
                _,
                _,
            ) = get_cu_seqblocks(
                cu_seqlens, max_seqlen_q, 1, block_size_k, seqlens_cpu
            )
    # Step 3: Sparse attention using topk index (main head). The MSA path only
    # replaces this step; the indexer above is unchanged. MSA has no attn-sink
    # input, so keep the Triton path when sink is present.
    if use_msa and sink is None:
        from .msa import MSAUnavailableError, msa_sparse_prefill_main

        try:
            o = msa_sparse_prefill_main(
                q=q_for_main,
                k_cache=k_cache,
                v_cache=v_cache,
                topk_idx=topk_idx,
                per_query_topk_idx=per_query_topk_idx,
                query_group_mask=query_group_mask,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                block_size_k=block_size_k,
                sm_scale=sm_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        except MSAUnavailableError as err:
            _warn_msa_fallback(err)
            o = flash_prefill_with_gqa_share_sparse(
                q=q_for_main,
                k_cache=k_cache,
                v_cache=v_cache,
                sink=sink,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                topk_idx=topk_idx,
                per_query_topk_idx=per_query_topk_idx,
                block_size_q=main_block_size_q,
                block_size_k=block_size_k,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                max_seqlen_q=max_seqlen_q,
                sm_scale=sm_scale,
                cu_seqblocks_q=main_cu_seqblocks_q,
                max_seqblock_q=main_max_seqblock_q,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                return_lse=dcp_size > 1,
            )
    else:
        o = flash_prefill_with_gqa_share_sparse(
            q=q_for_main,
            k_cache=k_cache,
            v_cache=v_cache,
            sink=sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            per_query_topk_idx=per_query_topk_idx,
            query_group_mask=query_group_mask,
            block_size_q=main_block_size_q,
            block_size_k=block_size_k,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale,
            cu_seqblocks_q=main_cu_seqblocks_q,
            max_seqblock_q=main_max_seqblock_q,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            return_lse=dcp_size > 1,
        )
    if dcp_size > 1:
        o, local_lse = o
        o = cp_lse_ag_out_rs_mha(
            o, local_lse, dcp_group, use_reduce_scatter=True
        )
    return idx_o, o


def minimax_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim], num_idx_heads >= num_kv_heads
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged); None when disable_index_value
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,  # max of seq_lens, passed from caller to avoid sync during CUDA graph capture
    block_size_q: int,  # useless for now, will always be 1
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[
        torch.Tensor
    ] = None,  # per-forward MSA page table (cached)
    msa_plan=None,  # per-forward MSA fmha_sm100 plan (cached)
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    dcp_group=None,
    verify_group_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_cache = _flatten_main_kv_cache(k_cache)
    v_cache = _flatten_main_kv_cache(v_cache)
    if dcp_size > 1:
        if dcp_size != 2:
            raise NotImplementedError(
                "MiniMax-M3 DCP currently supports only DCP2, matching its "
                "two-way replicated Main/Index KV heads"
            )
        if dcp_group is None or dcp_group.world_size != dcp_size:
            raise ValueError("MiniMax-M3 DCP2 requires its matching DCP group")
        if score_type != "max" or not disable_index_value:
            raise NotImplementedError(
                "MiniMax-M3 DCP2 requires score_type='max' and "
                "disable_index_value=True"
            )
        if dense_main_attn_fn is not None or use_msa:
            raise NotImplementedError(
                "MiniMax-M3 DCP2 currently supports only the Triton sparse "
                "main-attention path"
            )
        if sink is not None or idx_sink is not None:
            raise NotImplementedError("MiniMax-M3 DCP2 does not yet support sinks")
    # Step 1: Flash decode with topk index (using index head). When the dense main
    # attention is used, the indexer emits the page table directly (fused
    # transform) instead of block ids, plus the per-query effective KV length.
    idx_o, topk_idx, real_seq_lens = flash_decode_with_topk_idx(
        q=idx_q,
        sink=idx_sink,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        use_dense_main_attn=dense_main_attn_fn is not None,
        page_size=page_size,
        q_scale=idx_q_scale,
        k_scale=idx_k_scale,
        v_scale=idx_v_scale,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_group=dcp_group,
        verify_group_size=verify_group_size,
    )
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if dense_main_attn_fn is not None:
        # topk_idx is the page table; real_seq_lens is the per-query cache_seqlens
        assert idx_group_size == 1
        o = dense_main_attn_fn(q, topk_idx, real_seq_lens)
    else:
        # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )
        # Adjacent DCP2 ranks already replicate the same MiniMax KV head while
        # owning different Q-head shards. Gather those Q heads, attend only to
        # this rank's token shard, then merge the two partial softmaxes by LSE.
        q_for_main = (
            dcp_group.all_gather(q.contiguous(), dim=1).contiguous()
            if dcp_size > 1
            else q
        )
        # Step 3: Sparse attention using topk index (main head). The MSA path
        # only replaces this step; keep the Triton path when sink is present.
        if use_msa and sink is None:
            from .msa import MSAUnavailableError, msa_sparse_decode_main

            try:
                o = msa_sparse_decode_main(
                    q=q_for_main,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    topk_idx=topk_idx,
                    req_to_token=req_to_token,
                    slot_ids=slot_ids,
                    seq_lens=seq_lens,
                    block_size_k=block_size_k,
                    sm_scale=sm_scale,
                    kv_indices=msa_kv_indices,
                    plan=msa_plan,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            except MSAUnavailableError as err:
                _warn_msa_fallback(err)
                o = flash_decode_with_gqa_share_sparse(
                    q=q_for_main,
                    sink=sink,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    req_to_token=req_to_token,
                    seq_lens=seq_lens,
                    slot_ids=slot_ids,
                    block_size=block_size_k,
                    topk_idx=topk_idx,
                    sm_scale=sm_scale,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    return_lse=dcp_size > 1,
                )
        else:
            o = flash_decode_with_gqa_share_sparse(
                q=q_for_main,
                sink=sink,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                slot_ids=slot_ids,
                block_size=block_size_k,
                topk_idx=topk_idx,
                sm_scale=sm_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                return_lse=dcp_size > 1,
            )
        if dcp_size > 1:
            o, local_lse = o
            o = cp_lse_ag_out_rs_mha(
                o, local_lse, dcp_group, use_reduce_scatter=True
            )
    return idx_o, o
