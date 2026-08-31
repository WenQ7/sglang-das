"""Adapter from SGLang's FA3 call contract to the HCU FlashAttention package.

The BW1100 software image ships a gfx938-aware ``flash_attn`` build.  Its
``varlen_fwd_unified`` entry point accepts SGLang's ordinary BSHD paged KV
layout and handles both long-prefix extend and decode.  Keeping this adapter
small lets the existing :class:`FlashAttentionBackend` continue to own page
tables, CUDA-graph buffers, cache writes, and DP-attention metadata.

This module is intentionally selected only by the explicit ``hcu_fa`` backend.
It is not a silent replacement for CUDA FA3.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=3)
def _log_hcu_fa_selection(call_kind: str) -> None:
    logger.info(
        "[ATTN] HCU FlashAttention selected: call=%s, op=varlen_fwd_unified, layout=bshd",
        call_kind,
    )


@lru_cache(maxsize=1)
def _hcu_flash_ops():
    try:
        from flash_attn import flash_attn_varlen_func as raw_varlen
        from flash_attn import varlen_fwd_unified
    except ImportError as exc:  # pragma: no cover - depends on the HCU image
        raise RuntimeError(
            "hcu_fa requires the HCU flash_attn package with "
            "varlen_fwd_unified support"
        ) from exc
    return raw_varlen, varlen_fwd_unified


def _unsupported(name: str, value, default=None) -> None:
    if default is None:
        is_unsupported = value is not None
    else:
        is_unsupported = value != default
    if is_unsupported:
        raise NotImplementedError(f"hcu_fa does not support {name}={value!r}")


def _copy_out(result, out: Optional[torch.Tensor]):
    if out is None:
        return result
    if isinstance(result, tuple):
        out.copy_(result[0])
        return (out, *result[1:])
    out.copy_(result)
    return out


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk: Optional[int] = None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
    max_seqlen_k: Optional[int] = None,
):
    # SGLang writes K/V into the pool before invoking attention.  The fused
    # cache-update/rotary/FA4-only features are therefore outside this adapter.
    for name, value in (
        ("k", k),
        ("v", v),
        ("qv", qv),
        ("rotary_cos", rotary_cos),
        ("rotary_sin", rotary_sin),
        ("cache_batch_idx", cache_batch_idx),
        ("cache_leftpad", cache_leftpad),
        ("rotary_seqlens", rotary_seqlens),
        ("scheduler_metadata", scheduler_metadata),
        ("pack_gqa", pack_gqa),
        ("score_mod", score_mod),
        ("aux_tensors", aux_tensors),
        ("sfq", sfq),
        ("sfk", sfk),
        ("sfv", sfv),
        ("rel_bias", rel_bias),
        ("rel_bias_prep_cache", rel_bias_prep_cache),
    ):
        _unsupported(name, value)
    _unsupported("only_qv", only_qv, False)
    _unsupported("sm_margin", sm_margin, 0)
    if attention_chunk not in (None, 0):
        raise NotImplementedError(
            f"hcu_fa does not support attention_chunk={attention_chunk!r}"
        )
    if ver != 3:
        raise NotImplementedError(f"hcu_fa expects ver=3, got {ver}")
    if page_table is None or cache_seqlens is None or cu_seqlens_q is None:
        raise ValueError(
            "hcu_fa paged attention requires page_table, cache_seqlens, and "
            "cu_seqlens_q"
        )
    if not isinstance(cache_seqlens, torch.Tensor):
        cache_seqlens = torch.full(
            (cu_seqlens_q.numel() - 1,),
            int(cache_seqlens),
            dtype=torch.int32,
            device=q.device,
        )

    # FlashAttentionBackend already reshapes the ordinary SGLang pool to
    # [num_pages, page_size, num_kv_heads, head_dim] (BSHD).
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "hcu_fa expects 4D BSHD paged K/V caches, got "
            f"{tuple(k_cache.shape)} and {tuple(v_cache.shape)}"
        )
    if max_seqlen_q is None:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k is None:
        # Eager metadata slices page_table to the live maximum.  CUDA-graph
        # metadata uses the static context bound, which is also a valid upper
        # bound for the HCU kernel's scheduler.
        max_seqlen_k = page_table.shape[1] * k_cache.shape[1]

    _, unified = _hcu_flash_ops()
    _log_hcu_fa_selection("kvcache")
    return unified(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens.to(torch.int32),
        block_table=page_table.to(torch.int32),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        s_aux=sinks,
        layout="bshd",
        out=out,
        return_softmax_lse=return_softmax_lse,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
):
    for name, value in (
        ("seqused_q", seqused_q),
        ("qv", qv),
        ("pack_gqa", pack_gqa),
        ("score_mod", score_mod),
        ("aux_tensors", aux_tensors),
        ("sfq", sfq),
        ("sfk", sfk),
        ("sfv", sfv),
        ("rel_bias", rel_bias),
        ("rel_bias_prep_cache", rel_bias_prep_cache),
    ):
        _unsupported(name, value)
    _unsupported("only_qv", only_qv, False)
    _unsupported("sm_margin", sm_margin, 0)
    if attention_chunk not in (None, 0):
        raise NotImplementedError(
            f"hcu_fa does not support attention_chunk={attention_chunk!r}"
        )
    if ver != 3:
        raise NotImplementedError(f"hcu_fa expects ver=3, got {ver}")

    raw_varlen, unified = _hcu_flash_ops()
    if page_table is not None:
        if seqused_k is None:
            if cu_seqlens_k is None:
                raise ValueError("paged hcu_fa requires seqused_k or cu_seqlens_k")
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        _log_hcu_fa_selection("varlen_paged")
        result = unified(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k.to(torch.int32),
            block_table=page_table.to(torch.int32),
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            s_aux=sinks,
            layout="bshd",
            out=out,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return result

    _log_hcu_fa_selection("varlen_dense")
    result = raw_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=return_softmax_lse,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=sinks,
    )
    return _copy_out(result, out)
