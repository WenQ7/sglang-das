import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.utils import (
    get_dcp_cache_store_loc_and_mask,
    merge_dcp_block_scores,
)
from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mha
from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxHybridAttnBackend,
    _get_minimax_num_key_value_heads,
)


class _FakeDCPGroup:
    def __init__(self, peer_score: torch.Tensor):
        self.world_size = 2
        self.peer_score = peer_score

    def all_gather(self, local_score: torch.Tensor, dim: int):
        self.last_dim = dim
        return torch.cat([local_score, self.peer_score], dim=dim)


class _FakeLSEGroup:
    world_size = 2
    rank_in_group = 0

    def __init__(self, peer_out: torch.Tensor, peer_lse: torch.Tensor):
        self.peer_out = peer_out
        self.peer_lse = peer_lse
        self.local_lse = None

    def all_gather(self, local_lse: torch.Tensor, dim: int):
        self.local_lse = local_lse
        return torch.cat([local_lse, self.peer_lse], dim=dim)

    def all_reduce(self, local_weighted_out: torch.Tensor):
        lses = torch.stack([self.local_lse, self.peer_lse], dim=0)
        global_lse = torch.logsumexp(lses, dim=0)
        peer_scale = torch.exp(self.peer_lse - global_lse).unsqueeze(-1)
        peer_scale = torch.nan_to_num(peer_scale)
        return local_weighted_out + self.peer_out * peer_scale

    def reduce_scatter_along_dim(self, local_weighted_out: torch.Tensor, dim: int):
        assert dim == 1
        reduced = self.all_reduce(local_weighted_out)
        return reduced[:, : reduced.shape[1] // self.world_size].contiguous()


class TestMiniMaxDCP2Contract(unittest.TestCase):
    def test_hybrid_backend_exposes_shared_kv_pool_for_cp(self):
        kv_pool = object()
        req_pool = object()
        dense = SimpleNamespace(
            token_to_kv_pool=kv_pool,
            req_to_token_pool=req_pool,
            extend_dummy_seqs_capped_by_req_pool=False,
        )
        sparse = SimpleNamespace(
            token_to_kv_pool=kv_pool,
            extend_dummy_seqs_capped_by_req_pool=False,
        )

        backend = MiniMaxHybridAttnBackend(dense, sparse, [3])

        self.assertIs(backend.token_to_kv_pool, kv_pool)
        self.assertIs(backend.req_to_token_pool, req_pool)

    def test_hybrid_backend_rejects_different_child_kv_pools(self):
        dense = SimpleNamespace(token_to_kv_pool=object())
        sparse = SimpleNamespace(token_to_kv_pool=object())

        with self.assertRaisesRegex(RuntimeError, "same token_to_kv_pool"):
            MiniMaxHybridAttnBackend(dense, sparse, [3])

    def test_kv_heads_are_read_from_vl_text_config(self):
        text_config = SimpleNamespace(num_key_value_heads=4)
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
            hf_text_config=text_config,
        )

        self.assertEqual(_get_minimax_num_key_value_heads(model_config), 4)

    def test_kv_heads_flat_config_fallback(self):
        hf_config = SimpleNamespace(num_key_value_heads=8)
        model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=None)

        self.assertEqual(_get_minimax_num_key_value_heads(model_config), 8)

    def test_block_score_merge_is_exact_max(self):
        local = torch.tensor([[[-1.0, 4.0, float("-inf"), 7.0]]])
        peer = torch.tensor([[[3.0, 2.0, 5.0, float("-inf")]]])
        group = _FakeDCPGroup(peer)

        merged = merge_dcp_block_scores(local, group, 2)

        torch.testing.assert_close(merged, torch.tensor([[[3.0, 4.0, 5.0, 7.0]]]))
        self.assertEqual(group.last_dim, 0)

    def test_cache_store_uses_virtual_slot_division_and_position_owner(self):
        virtual_loc = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
        positions = torch.tensor([128, 129, 130, 131], dtype=torch.int64)

        loc0, mask0 = get_dcp_cache_store_loc_and_mask(
            virtual_loc, positions, None, dcp_size=2, dcp_rank=0
        )
        loc1, mask1 = get_dcp_cache_store_loc_and_mask(
            virtual_loc, positions, None, dcp_size=2, dcp_rank=1
        )

        torch.testing.assert_close(loc0, torch.tensor([10, 10, 11, 11]))
        torch.testing.assert_close(loc1, loc0)
        torch.testing.assert_close(mask0, torch.tensor([True, False, True, False]))
        torch.testing.assert_close(mask1, ~mask0)

    def test_cache_store_rejects_missing_mask_metadata(self):
        with self.assertRaisesRegex(RuntimeError, "ownership mask"):
            get_dcp_cache_store_loc_and_mask(
                torch.tensor([0, 1]),
                positions=None,
                fallback_mask=None,
                dcp_size=2,
                dcp_rank=0,
            )

    def test_output_lse_merge_handles_one_empty_rank(self):
        local_out = torch.tensor([[[2.0], [4.0], [6.0], [8.0]]])
        peer_out = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
        local_lse = torch.tensor([[0.0, float("-inf"), 1.0, float("-inf")]])
        peer_lse = torch.tensor([[float("-inf"), 0.0, float("-inf"), 1.0]])
        group = _FakeLSEGroup(peer_out, peer_lse)

        merged = cp_lse_ag_out_rs_mha(
            local_out, local_lse, group, use_reduce_scatter=True
        )

        # Rank 0 receives the first half of gathered Q heads. Head 0 comes
        # exclusively from local rank; head 1 exclusively from peer rank.
        torch.testing.assert_close(merged, torch.tensor([[[2.0], [20.0]]]))
        self.assertFalse(torch.isnan(merged).any())


if __name__ == "__main__":
    unittest.main()
