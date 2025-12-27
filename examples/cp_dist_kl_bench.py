"""Benchmark CP communication patterns on a single machine (CPU-only).

This script is meant to validate the *optimization idea* in
`codex/optimize-training-time-for-context-parallelism` locally:

- baseline: gather full log-prob sequences via `all_gather_with_cp`
  (implemented as an all-reduce over a `response_length` tensor)
- optimized: all-reduce only `(numerator, denominator)` scalars per sample

Run with local multi-process spawn (recommended on macOS):

  .venv/bin/python examples/cp_dist_kl_bench.py --nproc 2 \\
    --response-length 65536 --batch-size 32

Run with torchrun (may require a resolvable hostname):

  .venv/bin/torchrun --standalone --nproc_per_node=2 \\
    examples/cp_dist_kl_bench.py --response-length 65536 --batch-size 32

To A/B test a different worktree (e.g. main):

  SLIME_REPO_ROOT=../slime-main .venv/bin/torchrun --standalone --nproc_per_node=2 \\
    examples/cp_dist_kl_bench.py --response-length 65536 --batch-size 32
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import types
from pathlib import Path
from time import perf_counter

try:
    import torch
    import torch.distributed as dist
    import torch.distributed.nn  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for examples/cp_dist_kl_bench.py.\n"
        "Install it with the same interpreter you're using to run this script, e.g.:\n"
        "  python -m pip install torch\n"
    ) from exc


def _ensure_repo_on_syspath() -> None:
    repo_root = os.environ.get("SLIME_REPO_ROOT")
    resolved_root = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[1]
    if str(resolved_root) not in sys.path:
        sys.path.insert(0, str(resolved_root))


def _ensure_megatron_stub() -> None:
    """Provide a minimal Megatron stub so cp_utils can import offline."""

    if "megatron.core" in sys.modules:
        return

    megatron_module = types.ModuleType("megatron")
    megatron_module.__path__ = []  # Mark as a package for nested imports.
    megatron_core_module = types.ModuleType("megatron.core")

    class _MPUStub:
        def get_context_parallel_rank(self) -> int:
            raise RuntimeError("MPU stub should be overwritten by _bind_cp_mpu_to_dist().")

        def get_context_parallel_world_size(self) -> int:
            raise RuntimeError("MPU stub should be overwritten by _bind_cp_mpu_to_dist().")

        def get_context_parallel_group(self):
            raise RuntimeError("MPU stub should be overwritten by _bind_cp_mpu_to_dist().")

    megatron_core_module.mpu = _MPUStub()
    megatron_module.core = megatron_core_module

    sys.modules.setdefault("megatron", megatron_module)
    sys.modules.setdefault("megatron.core", megatron_core_module)


def _import_cp_utils():
    _ensure_repo_on_syspath()
    try:
        from slime.backends.megatron_utils import cp_utils  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        if not (getattr(exc, "name", "") or "").startswith("megatron"):
            raise
        _ensure_megatron_stub()
        from slime.backends.megatron_utils import cp_utils  # type: ignore[import-not-found]

    return cp_utils


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _init_dist(backend: str) -> None:
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend)


def _bind_cp_mpu_to_dist(cp_utils) -> None:
    cp_group = dist.group.WORLD

    cp_utils.mpu.get_context_parallel_rank = lambda: dist.get_rank()
    cp_utils.mpu.get_context_parallel_world_size = lambda: dist.get_world_size()
    cp_utils.mpu.get_context_parallel_group = lambda: cp_group


def _get_chunked_loss_mask(cp_utils, *, total_length: int, response_length: int, loss_mask: torch.Tensor) -> torch.Tensor:
    cp_size = cp_utils.mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return loss_mask

    if hasattr(cp_utils, "get_chunked_loss_masks"):
        chunked_masks, _ = cp_utils.get_chunked_loss_masks([total_length], [response_length], [loss_mask])
        return chunked_masks[0]

    prompt_length = total_length - response_length
    _, _, _, tokens_offset = cp_utils.get_logits_and_tokens_offset_with_cp(total_length, response_length)
    loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
    loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
    return torch.cat([loss_mask_0, loss_mask_1], dim=0)


def _time_per_step(fn, *, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    dist.barrier()

    start = perf_counter()
    for _ in range(steps):
        fn()
    dist.barrier()
    elapsed = perf_counter() - start

    return (elapsed / steps) * 1000 if steps else 0.0


def _reduce_stats(value_ms: float) -> tuple[float, float]:
    """Return (mean, max) over ranks."""

    t = torch.tensor(value_ms, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    mean = float(t.item()) / dist.get_world_size()

    t = torch.tensor(value_ms, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    vmax = float(t.item())
    return mean, vmax


def _run_worker(*, rank: int, world_size: int, init_method: str, backend: str, args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)

    cp_utils = _import_cp_utils()
    _bind_cp_mpu_to_dist(cp_utils)

    total_length = args.total_length
    response_length = args.response_length
    if total_length <= response_length:
        raise SystemExit("--total-length must be > --response-length (prompt length must be positive).")

    dtype = torch.float32
    loss_masks = [torch.ones(response_length, dtype=dtype) for _ in range(args.batch_size)]
    chunked_loss_masks = [
        _get_chunked_loss_mask(
            cp_utils, total_length=total_length, response_length=response_length, loss_mask=loss_mask
        )
        for loss_mask in loss_masks
    ]

    log_probs = [torch.randn(int(mask.numel()), dtype=dtype) for mask in chunked_loss_masks]
    old_log_probs = [torch.randn(int(mask.numel()), dtype=dtype) for mask in chunked_loss_masks]

    cp_group = cp_utils.mpu.get_context_parallel_group()

    def bench_gather() -> torch.Tensor:
        seq_kl_sum = torch.zeros((), dtype=dtype)
        for local_log_prob, local_old_log_prob, loss_mask in zip(log_probs, old_log_probs, loss_masks, strict=True):
            full_log_prob = cp_utils.all_gather_with_cp(local_log_prob, total_length, response_length)
            full_old_log_prob = cp_utils.all_gather_with_cp(local_old_log_prob, total_length, response_length)
            seq_kl_sum = seq_kl_sum + ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(
                loss_mask.sum(), 1
            )
        return seq_kl_sum

    def bench_reduce() -> torch.Tensor:
        pairs: list[torch.Tensor] = []
        for local_log_prob, local_old_log_prob, chunked_loss_mask in zip(
            log_probs, old_log_probs, chunked_loss_masks, strict=True
        ):
            local_num = ((local_old_log_prob - local_log_prob) * chunked_loss_mask).sum()
            local_den = chunked_loss_mask.sum()
            pairs.append(torch.stack((local_num, local_den)))

        stacked = torch.stack(pairs)
        reduced = dist.nn.functional.all_reduce(stacked, group=cp_group)
        seq_kls = reduced[:, 0] / torch.clamp_min(reduced[:, 1], 1)
        return seq_kls.sum()

    dist.barrier()
    if rank == 0:
        print(
            f"world_size={world_size}, total_length={total_length}, response_length={response_length}, batch_size={args.batch_size}"
        )

    if args.method in ("gather", "both"):
        gather_ms = _time_per_step(bench_gather, warmup=args.warmup, steps=args.steps)
        mean_ms, max_ms = _reduce_stats(gather_ms)
        if rank == 0:
            print(f"gather: mean {mean_ms:.3f} ms/step, max {max_ms:.3f} ms/step")

    if args.method in ("reduce", "both"):
        reduce_ms = _time_per_step(bench_reduce, warmup=args.warmup, steps=args.steps)
        mean_ms, max_ms = _reduce_stats(reduce_ms)
        if rank == 0:
            print(f"reduce: mean {mean_ms:.3f} ms/step, max {max_ms:.3f} ms/step")

    dist.destroy_process_group()


def _spawn_entry(rank: int, world_size: int, init_method: str, backend: str, args_dict: dict) -> None:
    args = argparse.Namespace(**args_dict)
    _run_worker(rank=rank, world_size=world_size, init_method=init_method, backend=backend, args=args)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only CP KL communication benchmark (torchrun required)")
    parser.add_argument("--backend", default="gloo", help="torch.distributed backend (default: gloo)")
    parser.add_argument("--nproc", type=int, default=2, help="Number of local processes when not using torchrun")
    parser.add_argument("--total-length", type=int, default=66048, help="Total length (prompt + response)")
    parser.add_argument("--response-length", type=int, default=65536, help="Response length")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of sequences per step")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--steps", type=int, default=50, help="Measured steps")
    parser.add_argument(
        "--method",
        choices=("gather", "reduce", "both"),
        default="both",
        help="Which communication pattern to benchmark",
    )
    args = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched by torchrun.
        _init_dist(args.backend)
        world_size = dist.get_world_size()
        if world_size <= 1:
            raise SystemExit("torchrun world_size must be > 1.")
        _run_worker(
            rank=dist.get_rank(),
            world_size=world_size,
            init_method="env://",
            backend=args.backend,
            args=args,
        )
        return

    # Spawn local processes (works even when hostname is not resolvable).
    world_size = int(args.nproc)
    if world_size <= 1:
        raise SystemExit("--nproc must be > 1.")

    port = _find_free_port()
    init_method = f"tcp://127.0.0.1:{port}"

    import torch.multiprocessing as mp

    mp.spawn(
        _spawn_entry,
        args=(world_size, init_method, args.backend, vars(args)),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
