import unittest

from sglang.srt.batch_overlap.operations import YieldOperation
from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _noop(*args, **kwargs):
    return None


class _FakeAttention:
    op_prepare = _noop
    op_core = _noop


class MiniMaxM3DecoderLayer:
    op_comm_prepare_attn = _noop
    op_gather_a = _noop
    op_gather_b = _noop
    op_mlp = _noop
    op_combine_a = _noop
    op_combine_b = _noop
    self_attn = _FakeAttention()


class TestMiniMaxM3OperationsStrategy(CustomTestCase):
    def test_decode_strategy_launches_and_waits_for_both_collectives(self):
        strategy = OperationsStrategy.init_new_tbo(
            [MiniMaxM3DecoderLayer()], ForwardMode.DECODE
        )
        self.assertEqual(strategy.tbo_delta_stages, 0)
        self.assertEqual(
            sum(isinstance(op, YieldOperation) for op in strategy.operations), 2
        )

    def test_extend_is_deliberately_not_registered(self):
        with self.assertRaises(NotImplementedError):
            OperationsStrategy.init_new_tbo(
                [MiniMaxM3DecoderLayer()], ForwardMode.EXTEND
            )


if __name__ == "__main__":
    unittest.main()
