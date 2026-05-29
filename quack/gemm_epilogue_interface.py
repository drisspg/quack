from collections.abc import Callable
import os
import weakref

import torch
from torch import Tensor

from quack._compile_payload import set_epilogue_source_cache_key
from quack.gemm_blockscaled_interface import mxfp8_scaled_mm_epilogue
from quack.gemm_act import gemm_act as gemm_act_dispatch
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    _select_varlen_n_config,
    _validate_local_reduce_op_and_dtype,
    gemm_act,
)


_VARLEN_N_CONFIG_CACHE: dict[
    int, tuple[weakref.ReferenceType[Tensor], int, tuple[int, int, int, int], GemmConfig, int]
] = {}


def _cached_varlen_n_config(a: Tensor, b: Tensor, offs: Tensor) -> tuple[GemmConfig, int]:
    shape_key = (a.shape[-2], a.shape[-1], b.shape[-1], b.device.index or 0)
    cached = _VARLEN_N_CONFIG_CACHE.get(id(offs))
    version = offs._version
    if cached is not None:
        cached_ref, cached_version, cached_shape_key, cached_config, cached_covered_n = cached
        if (
            cached_ref() is offs
            and cached_version == version
            and cached_shape_key == shape_key
        ):
            return cached_config, cached_covered_n

    offsets = [0, *(int(offset) for offset in offs.detach().cpu().tolist())]
    covered_n = offsets[-1]
    config = _select_varlen_n_config(a.device, a.shape[-2], a.shape[-1], offsets)
    if any(offset % config.tile_n != 0 for offset in offsets):
        raise NotImplementedError(
            "QUACK varlen-N currently requires cumulative N offsets to be tile_n-aligned; "
            "non-aligned partial N tiles must fall back to torch._grouped_mm"
        )
    _VARLEN_N_CONFIG_CACHE[id(offs)] = (weakref.ref(offs), version, shape_key, config, covered_n)
    return config, covered_n


def _cu_seqlens_from_offsets(offs: Tensor) -> Tensor:
    return torch.cat((offs.new_zeros(1), offs))


def _grouped_mm_3d_2d_epilogue(
    a: Tensor,
    b: Tensor,
    offs: Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    out_dtype,
    tuned: bool,
    epilogue_source: str | None,
) -> Tensor:
    config, covered_n = _cached_varlen_n_config(a, b, offs)
    if covered_n > b.shape[-1]:
        raise RuntimeError(
            f"grouped GEMM offsets must not exceed B's N dimension, got {covered_n} > {b.shape[-1]}"
        )
    postact_dtype = a.dtype if out_dtype is None else out_dtype
    cu_seqlens_n = _cu_seqlens_from_offsets(offs)
    if tuned:
        _, out = gemm_act(
            a,
            b,
            activation=None,
            store_preact=False,
            tuned=True,
            tensor_epilogue_fn=epilogue_fn,
            tensor_epilogue_key=epilogue_key,
            tensor_epilogue_source=epilogue_source,
            cu_seqlens_n=cu_seqlens_n,
            out_dtype=out_dtype,
            postact_dtype=postact_dtype,
        )
        return out
    out = torch.empty(
        (a.shape[-2], b.shape[-1]),
        device=a.device,
        dtype=postact_dtype,
    )
    gemm_act_dispatch(
        a,
        b.mT,
        None,
        None,
        out,
        None,
        None,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=config.is_dynamic_persistent,
        cu_seqlens_n=cu_seqlens_n,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
    )
    return out


def _infer_epilogue_arg_kind(a: Tensor, b: Tensor, arg: Tensor) -> str:
    m, n = a.shape[-2], b.shape[-1]
    if tuple(arg.shape) == (*a.shape[:-1], n):
        return "tile"
    if arg.shape == (1, n):
        return "row"
    if arg.shape == (m, 1):
        return "col"
    raise NotImplementedError(
        "QUACK captured tensor epilogue args currently must match the GEMM output "
        "shape or broadcast as [1, N] / [M, 1]"
    )


def _infer_epilogue_arg_kinds(
    a: Tensor,
    b: Tensor,
    epilogue_args: tuple[Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    inferred = tuple(_infer_epilogue_arg_kind(a, b, arg) for arg in epilogue_args)
    if epilogue_arg_kinds and epilogue_arg_kinds != inferred:
        raise RuntimeError(
            f"epilogue_arg_kinds={epilogue_arg_kinds} does not match inferred kinds {inferred!r}"
        )
    return inferred


def _validate_local_reduce(
    a: Tensor, b: Tensor, out: Tensor | None, group: int | None, dim: int | None
) -> int | None:
    if out is None:
        return None
    group = 32 if group is None else group
    if group <= 0 or group & (group - 1) != 0:
        raise NotImplementedError(
            f"QUACK local_reduce_out currently requires a positive power-of-two group, got {group}"
        )
    dim = 1 if dim is None else dim
    if dim == 1:
        reduce_size = b.shape[-1]
        expected = (*a.shape[:-1], reduce_size // group)
    elif dim == 0:
        if group > 16:
            raise NotImplementedError(
                "local M-group reductions currently support only group sizes <= 16"
            )
        reduce_size = a.shape[-2]
        if reduce_size % 128 != 0:
            raise NotImplementedError(
                "local M-group reductions currently require M to be a multiple of tile_m=128"
            )
        expected = (*a.shape[:-2], reduce_size // group, b.shape[-1])
    else:
        raise NotImplementedError(f"unsupported local_reduce_dim={dim}")
    if reduce_size % group != 0:
        raise RuntimeError(
            f"local_reduce_out requires reduced dim divisible by {group}, got {reduce_size}"
        )
    if tuple(out.shape) != tuple(expected):
        raise RuntimeError(f"local_reduce_out shape must be {expected}, got {tuple(out.shape)}")
    return group


def gemm_epilogue(
    a: Tensor,
    b: Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    C: Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    out_dtype=None,
    offs: Tensor | None = None,
    epilogue_args: tuple[Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    aux_out: Tensor | None = None,
    local_reduce_out: Tensor | None = None,
    local_reduce_group: int | None = None,
    local_reduce_dim: int | None = None,
    local_reduce_op: str = "sum",
    local_reduce_scale: float = 1.0,
    local_reduce_max_power: int = 8,
    local_reduce_feeds_main: bool = False,
    local_reduce_source_from_epilogue: bool = False,
    tuned: bool | None = None,
    epilogue_source: str | None = None,
    main_output_transform: str | None = None,
    main_output_transform_group: int | None = None,
    concat_layout: tuple[str, ...] | None = None,
) -> Tensor:
    if tuned is None:
        tuned = os.getenv("QUACK_GEMM_EPILOGUE_TUNED", "0") == "1"
    if epilogue_source is not None:
        set_epilogue_source_cache_key(epilogue_fn, epilogue_source)
    if local_reduce_out is not None:
        _validate_local_reduce_op_and_dtype(local_reduce_op, local_reduce_out)
        local_reduce_group = _validate_local_reduce(
            a, b, local_reduce_out, local_reduce_group, local_reduce_dim
        )
        local_reduce_dim = 1 if local_reduce_dim is None else local_reduce_dim
        if local_reduce_op != "sum" and local_reduce_dim != 1:
            raise NotImplementedError(
                "QUACK non-sum local_reduce_op currently supports only local N reductions"
            )
        if local_reduce_op in ("amax_abs", "mx_e8m0_scale", "nvfp4_e4m3_scale") and (
            local_reduce_group is None or local_reduce_group >= b.shape[-1]
        ):
            raise NotImplementedError(
                "QUACK non-sum local_reduce_op currently requires grouped local N reductions inside tile_N"
            )
        if local_reduce_op == "mx_e8m0_scale" and local_reduce_out.dtype is not torch.float8_e8m0fnu:
            raise NotImplementedError(
                "QUACK mx_e8m0_scale local_reduce_out must have dtype torch.float8_e8m0fnu"
            )
        if local_reduce_op == "nvfp4_e4m3_scale" and local_reduce_out.dtype is not torch.float8_e4m3fn:
            raise NotImplementedError(
                "QUACK nvfp4_e4m3_scale local_reduce_out must have dtype torch.float8_e4m3fn"
            )
    if aux_out is not None:
        if tuple(aux_out.shape) != (*a.shape[:-1], b.shape[-1]):
            raise RuntimeError(
                f"aux_out shape must match GEMM output shape {(*a.shape[:-1], b.shape[-1])}, "
                f"got {tuple(aux_out.shape)}"
            )
    if offs is not None:
        if (
            epilogue_args
            or epilogue_arg_kinds
            or aux_out is not None
            or local_reduce_out is not None
        ):
            raise NotImplementedError(
                "grouped GEMM epilogue does not support epilogue args, aux outputs, "
                "or local reductions yet"
            )
        if C is not None or scale_a is not None or scale_b is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError("QUACK grouped GEMM epilogue does not support C/scales/alpha/beta yet")
        if offs.dtype is not torch.int32:
            raise RuntimeError(f"grouped GEMM offsets must be int32, got {offs.dtype}")
        if a.dim() == 3 and b.dim() == 2:
            if main_output_transform is not None or main_output_transform_group is not None:
                raise NotImplementedError("QUACK varlen_n does not support shape-changing epilogues yet")
            if concat_layout:
                raise NotImplementedError("QUACK varlen_n does not support concat_layout epilogues yet")
            return _grouped_mm_3d_2d_epilogue(
                a,
                b,
                offs,
                epilogue_fn,
                epilogue_key,
                out_dtype,
                tuned,
                epilogue_source,
            )
        cu_seqlens = _cu_seqlens_from_offsets(offs)
        if a.dim() == 2 and b.dim() == 3:
            cu_seqlens_m = cu_seqlens
            cu_seqlens_k = None
            cu_seqlens_n = None
        elif a.dim() == 2 and b.dim() == 2:
            cu_seqlens_m = None
            cu_seqlens_k = cu_seqlens
            cu_seqlens_n = None
        else:
            raise NotImplementedError("QUACK grouped GEMM epilogue supports only 2D/3D, 2D/2D, and 3D/2D grouped_mm")
        _, out = gemm_act(
            a,
            b,
            activation=None,
            store_preact=False,
            tuned=tuned,
            tensor_epilogue_fn=epilogue_fn,
            tensor_epilogue_key=epilogue_key,
            tensor_epilogue_source=epilogue_source,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_n=cu_seqlens_n,
            out_dtype=out_dtype,
            postact_dtype=a.dtype if out_dtype is None else out_dtype,
            main_output_transform=main_output_transform,
            main_output_transform_group=main_output_transform_group,
            concat_layout=concat_layout,
        )
        return out
    if epilogue_arg_kinds and len(epilogue_arg_kinds) != len(epilogue_args):
        raise RuntimeError("epilogue_arg_kinds must match epilogue_args length")
    if epilogue_arg_kinds and not set(epilogue_arg_kinds) <= {"tile", "row", "col"}:
        raise NotImplementedError(
            f"QUACK GEMM epilogue supports only tile/row/col aux tensors, got {epilogue_arg_kinds}"
        )
    if epilogue_args:
        epilogue_arg_kinds = _infer_epilogue_arg_kinds(a, b, epilogue_args, epilogue_arg_kinds)
    if epilogue_args and C is not None:
        raise NotImplementedError("QUACK epilogue arg cannot be combined with C yet")
    if epilogue_args and (alpha != 1.0 or beta != 1.0):
        raise NotImplementedError(
            "QUACK epilogue args cannot be combined with non-default alpha/beta yet"
        )
    if main_output_transform is not None:
        if main_output_transform != "grouped_n_contract" or main_output_transform_group not in (2, 4):
            raise NotImplementedError(
                "QUACK shape-changing main epilogues currently support only "
                "grouped_n_contract groups 2 and 4, got "
                f"main_output_transform={main_output_transform!r}, "
                f"main_output_transform_group={main_output_transform_group!r}"
            )
        if b.shape[-1] % main_output_transform_group != 0:
            raise RuntimeError(
                "QUACK grouped_n_contract main output requires the GEMM N dimension "
                f"to be divisible by {main_output_transform_group}, got {b.shape[-1]}"
            )
        if epilogue_args or aux_out is not None or local_reduce_out is not None:
            raise NotImplementedError(
                "QUACK shape-changing main epilogues cannot be combined with aux outputs yet"
            )
        if C is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError(
                "QUACK shape-changing main epilogues do not support C/alpha/beta yet"
            )
        if scale_a is not None or scale_b is not None:
            raise NotImplementedError(
                "QUACK shape-changing main epilogues do not support scaled GEMM yet"
            )
        if local_reduce_feeds_main:
            raise NotImplementedError(
                "QUACK shape-changing main epilogues cannot be combined with local reductions yet"
            )
        _, out = gemm_act(
            a,
            b,
            activation=None,
            tuned=tuned,
            tensor_epilogue_fn=epilogue_fn,
            tensor_epilogue_key=epilogue_key,
            tensor_epilogue_source=epilogue_source,
            out_dtype=out_dtype,
            postact_dtype=a.dtype if out_dtype is None else out_dtype,
            store_preact=False,
            main_output_transform=main_output_transform,
            main_output_transform_group=main_output_transform_group,
            concat_layout=concat_layout,
        )
        return out
    if scale_a is not None or scale_b is not None:
        if local_reduce_out is not None:
            raise NotImplementedError("scaled GEMM epilogue local_reduce_out is not supported yet")
        if scale_a is None or scale_b is None:
            raise RuntimeError("scaled GEMM epilogue requires both scale_a and scale_b")
        if C is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError("scaled GEMM epilogue does not support C/alpha/beta yet")
        return mxfp8_scaled_mm_epilogue(
            a,
            b,
            scale_a,
            scale_b,
            epilogue_fn,
            epilogue_key,
            out_dtype=a.dtype if out_dtype is None else out_dtype,
        )
    if aux_out is not None and local_reduce_out is not None:
        raise NotImplementedError(
            "QUACK generic aux_out cannot be combined with local_reduce_out"
        )
    row_auxes = tuple(
        arg.squeeze(0)
        for arg, kind in zip(epilogue_args, epilogue_arg_kinds)
        if kind == "row"
    )
    col_auxes = tuple(
        arg.squeeze(-1)
        for arg, kind in zip(epilogue_args, epilogue_arg_kinds)
        if kind == "col"
    )
    tile_auxes = tuple(
        arg for arg, kind in zip(epilogue_args, epilogue_arg_kinds) if kind == "tile"
    )
    postact_dtype = a.dtype if out_dtype is None else out_dtype
    preact_out, out = gemm_act(
        a,
        b,
        C=C,
        bias=None,
        colvec_bias=None,
        activation=None,
        tuned=tuned,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_source=epilogue_source,
        tensor_epilogue_uses_c=bool(epilogue_args),
        tensor_epilogue_arg_kinds=epilogue_arg_kinds,
        tensor_epilogue_rowvec_biases=row_auxes,
        tensor_epilogue_colvec_biases=col_auxes,
        tensor_epilogue_tile_biases=tile_auxes,
        alpha=alpha,
        beta=beta,
        preact_out=None,
        postact_out=aux_out,
        out_dtype=out_dtype,
        postact_dtype=aux_out.dtype if aux_out is not None else postact_dtype,
        store_preact=aux_out is not None,
        tensor_epilogue_returns_aux=aux_out is not None,
        local_reduce_out=local_reduce_out,
        local_reduce_group=local_reduce_group,
        local_reduce_op=local_reduce_op,
        local_reduce_scale=local_reduce_scale,
        local_reduce_max_power=local_reduce_max_power,
        local_reduce_dim=local_reduce_dim,
        local_reduce_feeds_main=local_reduce_feeds_main,
        local_reduce_source_from_epilogue=local_reduce_source_from_epilogue,
    )
    return preact_out if aux_out is not None else out


def gemm_relu(a: Tensor, b: Tensor) -> Tensor:
    _, out = gemm_act(a, b, activation="relu", store_preact=False, tuned=False)
    return out
