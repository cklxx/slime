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


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline context parallel dry run")
    parser.add_argument("--total-length", type=int, default=16, help="Total length (prompt + response)")
    parser.add_argument("--response-length", type=int, default=8, help="Response length")
    parser.add_argument("--cp-size", type=int, default=2, help="Context parallel world size to mock")
    args = parser.parse_args()

    simulate_cp(total_length=args.total_length, response_length=args.response_length, cp_size=args.cp_size)


if __name__ == "__main__":
    main()
