# Copyright (C) 2026, QuACK team.
from functools import partial

import cutlass.cute as cute
import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import GemmConfig
from quack.gemm_epilogue_interface import gemm_epilogue
from quack.gemm_interface import gemm, gemm_ref, gemm_tuned

sm100_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="varlen-N grouped GEMM is implemented only for SM100/SM110",
)


@cute.jit
def relu_epilogue(acc):
    return cute.where(acc > cute.full_like(acc, 0), acc, cute.full_like(acc, 0))


def _make_case(seq_lens, *, m=256, k=64, extra_n=0, dtype=torch.bfloat16):
    device = "cuda"
    torch.manual_seed(0)
    num_groups = len(seq_lens)
    cu_seqlens_n = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    total_n = int(sum(seq_lens)) + extra_n
    A = torch.randn((num_groups, m, k), device=device, dtype=dtype)
    B = torch.randn((k, total_n), device=device, dtype=dtype)
    return A, B, cu_seqlens_n


def _run_varlen_n(A, B, cu_seqlens_n, *, tile_n=128):
    out = torch.empty((A.shape[-2], B.shape[-1]), device=A.device, dtype=A.dtype)
    config = GemmConfig(
        tile_m=128,
        tile_n=tile_n,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
        device_capacity=get_device_capacity(A.device)[0],
    )
    partial(gemm_tuned.fn, config=config)(A, B, out, cu_seqlens_n=cu_seqlens_n)
    return out


def _assert_matches_native_grouped_mm(A, B, cu_seqlens_n):
    out = gemm(A, B, cu_seqlens_n=cu_seqlens_n, tuned=True)
    ref = torch._grouped_mm(A, B, cu_seqlens_n[1:])
    covered_n = int(cu_seqlens_n[-1].item())
    torch.testing.assert_close(out[:, :covered_n], ref[:, :covered_n], atol=0.5, rtol=0.05)


def _assert_epilogue_matches_native_grouped_mm(A, B, cu_seqlens_n):
    out = gemm_epilogue(A, B, relu_epilogue, "test_varlen_n_relu", offs=cu_seqlens_n[1:])
    ref = torch._grouped_mm(A, B, cu_seqlens_n[1:]).relu()
    covered_n = int(cu_seqlens_n[-1].item())
    torch.testing.assert_close(out[:, :covered_n], ref[:, :covered_n], atol=0.5, rtol=0.05)


@sm100_only
def test_gemm_varlen_n_exact_tiles_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([128, 128, 128])
    _assert_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_gemm_varlen_n_exact_tiles_with_uncovered_tail_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([128, 128], extra_n=128)
    _assert_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_grouped_mm_epilogue_varlen_n_exact_tiles_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([128, 128])
    _assert_epilogue_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_grouped_mm_epilogue_varlen_n_uncovered_tail_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([128, 128], extra_n=128)
    _assert_epilogue_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_grouped_mm_epilogue_varlen_n_8_aligned_tiles_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([72, 184])
    _assert_epilogue_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_gemm_varlen_n_8_aligned_tiles_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([72, 184])
    out = _run_varlen_n(A, B, cu_seqlens_n, tile_n=8)
    ref = torch._grouped_mm(A, B, cu_seqlens_n[1:])
    torch.testing.assert_close(out, ref, atol=0.5, rtol=0.05)


@sm100_only
def test_gemm_varlen_n_32_aligned_partial_tiles_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([96, 160, 128])
    out = _run_varlen_n(A, B, cu_seqlens_n, tile_n=32)
    ref = torch._grouped_mm(A, B, cu_seqlens_n[1:])
    torch.testing.assert_close(out, ref, atol=0.5, rtol=0.05)


@sm100_only
def test_grouped_mm_epilogue_varlen_n_32_aligned_partial_tiles_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([96, 160, 128])
    _assert_epilogue_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_grouped_mm_epilogue_varlen_n_zero_width_groups_match_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([0, 128, 0], extra_n=64)
    _assert_epilogue_matches_native_grouped_mm(A, B, cu_seqlens_n)


@sm100_only
def test_gemm_varlen_n_non_native_aligned_tiles_reject_before_launch():
    A, B, cu_seqlens_n = _make_case([65, 191])
    with pytest.raises(NotImplementedError, match="tile_n-aligned"):
        _run_varlen_n(A, B, cu_seqlens_n, tile_n=8)


@sm100_only
def test_grouped_mm_epilogue_varlen_n_non_native_aligned_tiles_reject_before_launch():
    A, B, cu_seqlens_n = _make_case([65, 191])
    with pytest.raises(NotImplementedError, match="tile_n-aligned"):
        gemm_epilogue(A, B, relu_epilogue, "test_varlen_n_relu", offs=cu_seqlens_n[1:])


@sm100_only
def test_grouped_mm_epilogue_varlen_n_offsets_beyond_total_n_reject_before_launch():
    A, B, cu_seqlens_n = _make_case([128, 128])
    offs = cu_seqlens_n[1:].clone()
    offs[-1] = B.shape[-1] + 128
    with pytest.raises(RuntimeError, match="must not exceed"):
        gemm_epilogue(A, B, relu_epilogue, "test_varlen_n_relu", offs=offs)


@sm100_only
def test_gemm_ref_varlen_n_matches_native_grouped_mm():
    A, B, cu_seqlens_n = _make_case([96, 160, 128], extra_n=128)
    out = torch.zeros((A.shape[-2], B.shape[-1]), device=A.device, dtype=torch.float32)
    ref = torch._grouped_mm(A.float(), B.float(), cu_seqlens_n[1:])
    gemm_ref(A.float(), B.float(), out=out, cu_seqlens_n=cu_seqlens_n)
    covered_n = int(cu_seqlens_n[-1].item())
    torch.testing.assert_close(out[:, :covered_n], ref[:, :covered_n])
