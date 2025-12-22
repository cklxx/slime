"""Offline context parallel dry-run helper.

This script demonstrates how context parallel (CP) chunking behaves
without launching the full distributed training stack. It mocks the
Megatron-LM context parallel APIs and reduces the distributed collectives
to simple local operations so that you can quickly inspect the offsets
and logits stitching logic on a CPU-only host.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from contextlib import contextmanager
from time import perf_counter
from unittest.mock import patch

import torch

from slime.backends.megatron_utils import cp_utils


@contextmanager
def _mock_cp_env(cp_rank: int, cp_size: int):
    """Mock the minimal CP APIs used by :mod:`cp_utils`.

    The all-reduce is replaced with an identity function because we only
    want to visualize how tensors are padded and reassembled locally.
    """

    def _identity_all_reduce(tensor: torch.Tensor, group=None):  # type: ignore[override]
        del group
        return tensor

    with (
        patch("slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_rank", return_value=cp_rank),
        patch("slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_world_size", return_value=cp_size),
        patch("slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_group", return_value=None),
        patch("slime.backends.megatron_utils.cp_utils.dist.nn.all_reduce", _identity_all_reduce),
    ):
        yield


def _format_range(r: tuple[int, int]) -> str:
    return f"[{r[0]}, {r[1]})"


def _pretty_offsets(title: str, ranges: Iterable[tuple[int, int]]) -> str:
    ranges_str = ", ".join(_format_range(r) for r in ranges)
    return f"{title}: {ranges_str}"


def _timeit(fn, *, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()

    total = 0.0
    for _ in range(steps):
        start = perf_counter()
        fn()
        total += perf_counter() - start

    return total / steps if steps else 0.0


def simulate_cp(total_length: int, response_length: int, cp_size: int) -> None:
    print(f"Simulating CP with total_length={total_length}, response_length={response_length}, cp_size={cp_size}\n")
    loss_mask = torch.ones(response_length)

    for cp_rank in range(cp_size):
        with _mock_cp_env(cp_rank, cp_size):
            chunk_size, chunks, logits_offset, token_offset = cp_utils.get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            print(f"# CP rank {cp_rank}")
            print(f"chunk_size: {chunk_size}")
            print(_pretty_offsets("chunks", chunks))
            print(_pretty_offsets("logits", logits_offset))
            print(_pretty_offsets("tokens", token_offset))

            chunked_masks, chunk_lengths = cp_utils.get_chunked_loss_masks(
                [total_length], [response_length], [loss_mask]
            )
            print(f"chunk_lengths: {chunk_lengths}")
            print(f"local loss mask: {chunked_masks[0].tolist()}")

            fake_logits = torch.arange(sum(chunk_lengths), dtype=torch.float32)
            gathered = cp_utils.all_gather_with_cp(fake_logits, total_length, response_length)
            print(f"gathered logits shape: {tuple(gathered.shape)}")
            print(f"gathered logits: {gathered.tolist()}\n")


def benchmark_cp(
    total_length: int,
    response_length: int,
    warmup: int,
    steps: int,
    cp_sizes: Iterable[int] = (1, 2),
) -> None:
    """Compare per-rank iteration time between CP=1 (baseline) and higher CP sizes."""

    loss_mask = torch.ones(response_length)

    def run_once(cp_rank: int, cp_size: int) -> None:
        with _mock_cp_env(cp_rank, cp_size):
            chunked_masks, chunk_lengths = cp_utils.get_chunked_loss_masks(
                [total_length], [response_length], [loss_mask]
            )
            chunk_len_sum = int(sum(chunk_lengths))
            fake_logits = torch.randn(chunk_len_sum, dtype=torch.float32)
            gathered = cp_utils.all_gather_with_cp(fake_logits, total_length, response_length)
            _ = gathered.sum()

    for cp_size in cp_sizes:
        per_rank: list[float] = []
        for cp_rank in range(cp_size):
            per_rank.append(
                _timeit(lambda cr=cp_rank, cs=cp_size: run_once(cr, cs), warmup=warmup, steps=steps)
            )

        avg_ms = sum(per_rank) * 1000 / len(per_rank)
        print(
            f"CP size {cp_size}: average per-rank step time over {steps} steps "
            f"(after {warmup} warmup) = {avg_ms:.3f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline context parallel dry run")
    parser.add_argument("--total-length", type=int, default=16, help="Total length (prompt + response)")
    parser.add_argument("--response-length", type=int, default=8, help="Response length")
    parser.add_argument("--cp-size", type=int, default=2, help="Context parallel world size to mock")
    parser.add_argument("--benchmark", action="store_true", help="Compare runtime for CP=1 vs CP=2")
    parser.add_argument("--benchmark-steps", type=int, default=50, help="Benchmark iterations per rank")
    parser.add_argument("--benchmark-warmup", type=int, default=10, help="Warmup iterations per rank")
    args = parser.parse_args()

    simulate_cp(total_length=args.total_length, response_length=args.response_length, cp_size=args.cp_size)

    if args.benchmark:
        print("Running CP runtime benchmark...\n")
        benchmark_cp(
            total_length=args.total_length,
            response_length=args.response_length,
            warmup=args.benchmark_warmup,
            steps=args.benchmark_steps,
        )


if __name__ == "__main__":
    main()
