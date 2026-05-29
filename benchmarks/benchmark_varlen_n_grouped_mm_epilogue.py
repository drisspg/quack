#!/usr/bin/env python3
"""Benchmark 3D x 2D grouped-mm epilogue dynamic-N paths."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import cutlass.cute as cute
import torch
from torch._higher_order_ops import grouped_mm_epilogue
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_in_microseconds

from quack.gemm_act import gemm_act as gemm_act_dispatch
from quack.gemm_epilogue_interface import gemm_epilogue


@cute.jit
def relu_epilogue(acc):
    return cute.where(acc > cute.full_like(acc, 0), acc, cute.full_like(acc, 0))


@dataclass(frozen=True)
class BenchmarkCase:
    groups: int
    m: int
    k: int
    n_per_group: int
    extra_n: int
    dtype: torch.dtype

    @property
    def total_n(self) -> int:
        return self.groups * self.n_per_group + self.extra_n


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    latency_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n-per-group", type=int, default=128)
    parser.add_argument("--extra-n", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run a small built-in tile-aligned shape sweep instead of one shape",
    )
    parser.add_argument("--compile", action="store_true", help="include torch.compile variants")
    parser.add_argument(
        "--cuda-graphs", action="store_true", help="time CUDA graph replay where possible"
    )
    parser.add_argument("--no-direct-quack", action="store_true")
    parser.add_argument("--no-preallocated-quack", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def make_inputs(case: BenchmarkCase):
    torch.manual_seed(0)
    a = torch.randn((case.groups, case.m, case.k), device="cuda", dtype=case.dtype)
    b = torch.randn((case.k, case.total_n), device="cuda", dtype=case.dtype)
    offs = torch.arange(
        case.n_per_group,
        case.groups * case.n_per_group + 1,
        case.n_per_group,
        device="cuda",
        dtype=torch.int32,
    )
    return a, b, offs


def native_grouped_mm_relu(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return torch._grouped_mm(a, b, offs).relu()


def make_unfused_loop_relu(offs: torch.Tensor) -> Callable:
    offsets = offs.tolist()

    def fn(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        out = torch.empty((a.shape[1], b.shape[1]), device=a.device, dtype=a.dtype)
        prev = 0
        for group, end in enumerate(offsets):
            out[:, prev:end] = a[group] @ b[:, prev:end]
            prev = end
        return out.relu()

    return fn


def quack_direct_relu(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return gemm_epilogue(a, b, relu_epilogue, "bench_varlen_n_relu", offs=offs, tuned=False)


def select_tile_n(offs: torch.Tensor) -> int:
    offsets = offs.tolist()
    for tile_n in (64, 32, 16, 8):
        if all(offset % tile_n == 0 for offset in offsets):
            return tile_n
    raise RuntimeError("current QUACK varlen-N benchmark requires 8-aligned group N")


def make_preallocated_quack_relu(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> Callable:
    cu_seqlens_n = torch.empty(offs.shape[0] + 1, device=offs.device, dtype=offs.dtype)
    cu_seqlens_n[0] = 0
    cu_seqlens_n[1:] = offs
    out = torch.empty((a.shape[1], b.shape[1]), device=a.device, dtype=a.dtype)
    tile_n = select_tile_n(offs)

    def fn(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        gemm_act_dispatch(
            a,
            b.mT,
            None,
            None,
            out,
            None,
            None,
            128,
            tile_n,
            1,
            1,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=False,
            cu_seqlens_n=cu_seqlens_n,
            tensor_epilogue_fn=relu_epilogue,
            tensor_epilogue_key="bench_varlen_n_relu",
        )
        return out

    return fn


def compiled_grouped_epilogue(backend: str) -> Callable:
    def fn(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        return grouped_mm_epilogue(
            a,
            b,
            lambda acc: acc.relu(),
            offs=offs,
            kernel_options={"backend": backend},
        )

    return torch.compile(fn, backend="inductor", fullgraph=True)


def check_correctness(
    name: str, fn: Callable, a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> None:
    actual = fn(a, b, offs)
    expected = native_grouped_mm_relu(a, b, offs)
    covered_n = int(offs[-1].item())
    torch.testing.assert_close(
        actual[:, :covered_n],
        expected[:, :covered_n],
        atol=0.5,
        rtol=0.05,
        msg=lambda msg: f"{name} mismatch\n{msg}",
    )


def benchmark_one(
    name: str, fn: Callable, a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, args
) -> BenchmarkResult:
    fn(a, b, offs)
    torch.cuda.synchronize()
    latency_us = benchmark_cuda_function_in_microseconds(
        fn,
        a,
        b,
        offs,
        NUM_ITERS=args.iters,
        USE_CUDA_GRAPHS=args.cuda_graphs,
    )
    return BenchmarkResult(name, latency_us)


def benchmark_case(args: argparse.Namespace, case: BenchmarkCase) -> list[BenchmarkResult]:
    if case.n_per_group % 8 != 0:
        raise RuntimeError("current QUACK varlen-N benchmark requires 8-aligned group N")

    a, b, offs = make_inputs(case)
    variants: list[tuple[str, Callable]] = [
        ("torch._grouped_mm + relu", native_grouped_mm_relu),
        ("unfused Python loop mm slices + relu", make_unfused_loop_relu(offs)),
    ]
    if not args.no_direct_quack:
        variants.append(("QuACK direct grouped_mm_epilogue", quack_direct_relu))
    if not args.no_preallocated_quack:
        variants.append(("QuACK preallocated grouped_mm_epilogue", make_preallocated_quack_relu(a, b, offs)))
    if args.compile:
        variants.extend(
            [
                ("Inductor TRITON grouped_mm_epilogue", compiled_grouped_epilogue("TRITON")),
                ("Inductor QUACK grouped_mm_epilogue", compiled_grouped_epilogue("QUACK")),
            ]
        )

    print(
        "measurement_contract="
        f"shape=G{case.groups}_M{case.m}_K{case.k}_Npg{case.n_per_group}_extraN{case.extra_n}, "
        f"dtype={case.dtype}, timing=transformer_nuggets.benchmark_cuda_function_in_microseconds, "
        f"iters={args.iters}, cuda_graphs={args.cuda_graphs}, allocation=variant-dependent"
    )
    results = []
    for name, fn in variants:
        if not args.skip_correctness:
            check_correctness(name, fn, a, b, offs)
        result = benchmark_one(name, fn, a, b, offs, args)
        results.append(result)

    native_us = results[0].latency_us
    for result in results:
        speedup = native_us / result.latency_us
        print(f"{result.name:42s} {result.latency_us:9.2f} us  speedup_vs_native={speedup:6.3f}x")
    return results


def sweep_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(groups=2, m=64, k=64, n_per_group=128, extra_n=0, dtype=torch.bfloat16),
        BenchmarkCase(groups=4, m=128, k=128, n_per_group=128, extra_n=0, dtype=torch.bfloat16),
        BenchmarkCase(groups=8, m=256, k=256, n_per_group=128, extra_n=0, dtype=torch.bfloat16),
        BenchmarkCase(groups=8, m=512, k=256, n_per_group=128, extra_n=0, dtype=torch.bfloat16),
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() < (10, 0):
        raise RuntimeError("SM100+ is required for the QUACK 3D x 2D varlen-N path")

    cases = (
        sweep_cases()
        if args.sweep
        else [
            BenchmarkCase(
                groups=args.groups,
                m=args.m,
                k=args.k,
                n_per_group=args.n_per_group,
                extra_n=args.extra_n,
                dtype=torch.bfloat16,
            )
        ]
    )
    for case in cases:
        benchmark_case(args, case)


if __name__ == "__main__":
    main()
