from __future__ import annotations

import atexit
import contextlib
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import get_benchmark_range, multigpu_bench_main
from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)
register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
DTYPE_ITEMSIZE = DTYPE.itemsize
MESSAGE_SIZES_KB = [2**x for x in range(2, 19)]
# Decode-sized MiniMax-M3 hidden states plus the important long-extend
# collectives. 16K x 6144 x bf16 is exactly 192 MiB.
MESSAGE_SIZES_KB += [12, 24, 48, 96, 192, 384, 640, 768, 896, 1536, 3072, 98304, 196608]
MESSAGE_SIZES_KB.sort()
if os.environ.get("SGLANG_BENCH_AR_MESSAGE_KB"):
    MESSAGE_SIZES_KB = [
        int(value)
        for value in os.environ["SGLANG_BENCH_AR_MESSAGE_KB"].split(",")
        if value.strip()
    ]
WORLD_SIZES = list(range(2, 9)) + [16]
MAX_BYTES = max(MESSAGE_SIZES_KB) * 1024
# trtllm allreduce_fusion only supports these world sizes.
FI_SUPPORTED_WORLD_SIZES = (2, 4, 8)
QR_MIN_BYTES = {
    # BF16, no cast.
    "qr-fp": {2: 2 * 1024**2, 4: 8 * 1024**2, 8: 16 * 1024**2},
    # INT4 is intentionally paired with BF16->FP16: the BF16 world-size=8
    # threshold is 2 GiB and would never select MiniMax's 192 MiB collective.
    "qr-int4": {2: 1 * 1024**2, 4: 2 * 1024**2, 8: 2 * 1024**2},
}
# AOT custom_all_reduce (v1) only supports these world sizes.
AOT_SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
# jit-eager times the naive-loop dispatch (eager heuristics); jit-graph
# captures the calls in a CUDA graph (graph heuristics + pointer table).
ALL_PROVIDERS = [
    "nccl",
    "aot",
    "jit-eager",
    "jit-graph",
    "fi",
    "qr-fp",
    "qr-int4",
]
# Keep the lossy path opt-in; the performance sweep requests it explicitly and
# runs it in a separate process so two 512 MiB QuickReduce IPC workspaces do
# not coexist in the generic all-provider benchmark.
PROVIDERS = [name for name in ALL_PROVIDERS if name != "qr-int4"]
if os.environ.get("SGLANG_BENCH_AR_PROVIDERS"):
    requested_providers = [
        value.strip()
        for value in os.environ["SGLANG_BENCH_AR_PROVIDERS"].split(",")
        if value.strip()
    ]
    unknown_providers = set(requested_providers) - set(ALL_PROVIDERS)
    if unknown_providers:
        raise ValueError(f"Unknown all-reduce providers: {sorted(unknown_providers)}")
    PROVIDERS = requested_providers
WORLD_SIZES = get_benchmark_range(WORLD_SIZES, [2, 4, 8])

# ---------------------------------------------------------------------------
# Per-rank distributed init (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group() -> dist.ProcessGroup:
    _init_cpu_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_group = torch.distributed.new_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    assert isinstance(device_group, dist.ProcessGroup)
    return device_group


# ---------------------------------------------------------------------------
# Backend wrappers - each exposes:
#   .all_reduce(tensor) -> Tensor
#   .graph_context()    -> context manager wrapping cuda-graph capture
#                          (nullcontext when capture is not required)
# ---------------------------------------------------------------------------


class NCCLAllReduceBackend:
    def __init__(self) -> None:
        self.group = _init_nccl_group()

    def graph_context(self):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self.group)
        return tensor


class JITAllReduceBackend:
    def __init__(self) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        # tuned workspace sizes, capped at the sweep maximum
        self.comm = CustomAllReduceV2(_init_cpu_group(), device, max_size=MAX_BYTES)
        if self.comm.disabled:
            raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
        # keep the whole sweep on the custom-AR path: the tuned config would
        # otherwise send the largest sizes back to NCCL
        self.comm.uncap_pull_thresholds()
        register_comm_cleanup(self.comm)

    def graph_context(self):
        return self.comm.capture()

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class AOTAllReduceBackend:
    def __init__(self) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.comm = CustomAllreduce(_init_cpu_group(), device, max_size=MAX_BYTES)
        if self.comm.disabled:
            raise RuntimeError("AOT CustomAllreduce is disabled on this system")
        register_comm_cleanup(self.comm)

    def graph_context(self):
        return self.comm.capture()

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class FlashInferAllReduceBackend:
    def __init__(self) -> None:
        import flashinfer.comm as comm

        group = _init_cpu_group()
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        # Use the smallest message size as the inner hidden dim, so any
        # message in the sweep is an integer multiple of it.
        hidden_dim = 1024 * min(MESSAGE_SIZES_KB) // DTYPE_ITEMSIZE
        num_tokens = MAX_BYTES // (hidden_dim * DTYPE_ITEMSIZE)
        self._comm = comm
        self._hidden_dim = hidden_dim
        self._workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=num_tokens,
            hidden_dim=hidden_dim,
            dtype=DTYPE,
        )

    def graph_context(self):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._comm.allreduce_fusion(
            input=tensor.view(-1, self._hidden_dim),
            workspace=self._workspace,
            pattern=self._comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=is_arch_support_pdl(),
            fp32_acc=True,
        )


class QuickReduceBackend:
    """QuickReduce candidate for BW1100 long-prefill traffic."""

    def __init__(self, regime: str, cast_bf16_to_fp16: bool) -> None:
        self.regime = regime
        self.cast_bf16_to_fp16 = cast_bf16_to_fp16
        self.label = f"{regime}/cast{int(cast_bf16_to_fp16)}"
        os.environ["SGLANG_ALLOW_GFX938_QUICK_ALLREDUCE"] = "1"
        os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = regime
        os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = str(
            int(cast_bf16_to_fp16)
        )
        os.environ.setdefault("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "512")

        from sglang.srt.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self.comm = QuickAllReduce(_init_cpu_group(), device)
        if self.comm.disabled:
            raise RuntimeError(f"QuickReduce {self.label} is disabled on this system")
        register_comm_cleanup(self.comm)
        if os.environ.get("SGLANG_BENCH_QR_VALIDATE", "1") == "1":
            self._validate_against_nccl()

    def _validate_against_nccl(self) -> None:
        """Validate the threshold size and MiniMax's 16K x 6144 message."""
        device_id = int(os.environ["LOCAL_RANK"])
        gpu_group = _init_nccl_group()
        for message_bytes in (16 * 1024**2, 192 * 1024**2):
            numel = message_bytes // DTYPE_ITEMSIZE
            generator = torch.Generator(device=f"cuda:{device_id}")
            generator.manual_seed(20260828 + device_id)
            inp = torch.randn(
                numel,
                dtype=DTYPE,
                device=f"cuda:{device_id}",
                generator=generator,
            )
            ref = inp.clone()
            dist.all_reduce(ref, group=gpu_group)
            assert self.comm.should_quick_allreduce(inp), message_bytes
            out = self.comm.quick_all_reduce(inp)
            torch.cuda.synchronize(device_id)
            max_abs = (out.float() - ref.float()).abs().max().item()
            error = out.float() - ref.float()
            rmse = error.square().mean().sqrt().item()
            ref_rms = ref.float().square().mean().sqrt().item()
            nrmse = rmse / max(ref_rms, 1e-12)
            if self.regime == "FP":
                valid = torch.allclose(out, ref, rtol=2e-2, atol=2e-2)
            else:
                # Quantized communication is accuracy-gated at model level.
                # Here reject gross corruption/non-finite output while reporting
                # NRMSE so the benchmark artifact records its numerical cost.
                valid = torch.isfinite(out).all().item() and nrmse <= 0.25
            if not valid:
                raise AssertionError(
                    f"QuickReduce {self.label} mismatch against NCCL: "
                    f"bytes={message_bytes}, max_abs={max_abs}, nrmse={nrmse}"
                )
            if device_id == 0:
                print(
                    f"QuickReduce {self.label} correctness OK: "
                    f"bytes={message_bytes}, max_abs={max_abs:.6g}, "
                    f"nrmse={nrmse:.6g}"
                )

    def graph_context(self):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.comm.should_quick_allreduce(tensor), str(tensor.shape)
        return self.comm.quick_all_reduce(tensor)


@cache_once
def _init_nccl_backend() -> NCCLAllReduceBackend:
    return NCCLAllReduceBackend()


@cache_once
def _init_jit_backend() -> JITAllReduceBackend:
    return JITAllReduceBackend()


@cache_once
def _init_aot_backend() -> AOTAllReduceBackend:
    return AOTAllReduceBackend()


@cache_once
def _init_fi_backend() -> FlashInferAllReduceBackend:
    return FlashInferAllReduceBackend()


@cache_once
def _init_qr_fp_backend() -> QuickReduceBackend:
    return QuickReduceBackend("FP", False)


@cache_once
def _init_qr_int4_backend() -> QuickReduceBackend:
    return QuickReduceBackend("INT4", True)


BACKEND_FACTORY = {
    "nccl": _init_nccl_backend,
    "jit-eager": _init_jit_backend,
    "jit-graph": _init_jit_backend,
    "aot": _init_aot_backend,
    "fi": _init_fi_backend,
    "qr-fp": _init_qr_fp_backend,
    "qr-int4": _init_qr_int4_backend,
}


def _quick_reduce_supported() -> bool:
    if not torch.version.hip:
        return False
    os.environ["SGLANG_ALLOW_GFX938_QUICK_ALLREDUCE"] = "1"
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        qr_rocm_arch_available,
    )

    return qr_rocm_arch_available()


@cache_once
def _init_all_backends() -> None:
    """Pre-build every supported backend before any timed iteration so JIT
    compilation / IPC setup don't bleed into the first measured size.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:  # NOTE: log some verbose info on initialization
        logging.basicConfig(level=logging.INFO)

    world_size = dist.get_world_size(_init_cpu_group())
    factories = {name: BACKEND_FACTORY[name] for name in PROVIDERS}
    if world_size not in AOT_SUPPORTED_WORLD_SIZES:
        factories.pop("aot", None)
    if world_size not in FI_SUPPORTED_WORLD_SIZES:
        factories.pop("fi", None)
    if world_size not in FI_SUPPORTED_WORLD_SIZES or not _quick_reduce_supported():
        for provider in QR_MIN_BYTES:
            factories.pop(provider, None)
    for fn in factories.values():
        fn()

    # reset level to warning
    logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@marker.parametrize("message_KB", MESSAGE_SIZES_KB)
@marker.benchmark("provider", PROVIDERS)
def benchmark(message_KB: int, provider: str):
    cpu_group = _init_cpu_group()
    gpu_group = _init_nccl_group()
    world_size = dist.get_world_size(cpu_group)
    if provider == "fi" and world_size not in FI_SUPPORTED_WORLD_SIZES:
        marker.skip(
            f"flashinfer trtllm allreduce_fusion needs world_size in "
            f"{FI_SUPPORTED_WORLD_SIZES}"
        )
    if provider == "aot" and world_size not in AOT_SUPPORTED_WORLD_SIZES:
        marker.skip(
            f"AOT custom_all_reduce needs world_size in " f"{AOT_SUPPORTED_WORLD_SIZES}"
        )
    if provider in QR_MIN_BYTES and (
        world_size not in FI_SUPPORTED_WORLD_SIZES or not _quick_reduce_supported()
    ):
        marker.skip("QuickReduce needs ROCm and world_size in (2, 4, 8)")
    message_bytes = message_KB * 1024
    if provider in QR_MIN_BYTES and message_bytes < QR_MIN_BYTES[provider][world_size]:
        marker.skip(
            f"{provider} minimum for world_size={world_size} is "
            f"{QR_MIN_BYTES[provider][world_size]} bytes"
        )
    _init_all_backends()
    backend = BACKEND_FACTORY[provider]()
    numel = message_bytes // DTYPE_ITEMSIZE
    device_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{device_id}")
    x = torch.randn(numel, dtype=DTYPE, device=device)
    ctx_fn = backend.graph_context if not provider.endswith("eager") else None
    # Bandwidth-equivalent bytes moved by a ring all-reduce per rank.
    effective_bytes = int(x.nbytes * 2 * (world_size - 1) / world_size)
    return marker.do_bench(
        backend.all_reduce,
        input_args=(x,),
        graph_context_fn=ctx_fn,
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        # all-reduce is in-place w.r.t. its argument; explicit footprint
        # captures the cross-GPU traffic instead.
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=effective_bytes,
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=WORLD_SIZES,
        main_fn=benchmark.run,
    )
