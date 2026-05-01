from collections.abc import Callable

import torch
from torch import Tensor

from quack.gemm_blockscaled_interface import mxfp8_scaled_mm_epilogue
from quack.gemm_interface import gemm_act


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
) -> Tensor:
    if offs is not None:
        if C is not None or scale_a is not None or scale_b is not None or alpha != 1.0 or beta != 1.0:
            raise NotImplementedError("QUACK grouped GEMM epilogue does not support C/scales/alpha/beta yet")
        if offs.dtype is not torch.int32:
            raise RuntimeError(f"grouped GEMM offsets must be int32, got {offs.dtype}")
        cu_seqlens = torch.empty(offs.shape[0] + 1, device=offs.device, dtype=offs.dtype)
        cu_seqlens[0] = 0
        cu_seqlens[1:] = offs
        if a.dim() == 2 and b.dim() == 3:
            cu_seqlens_m = cu_seqlens
            cu_seqlens_k = None
        elif a.dim() == 2 and b.dim() == 2:
            cu_seqlens_m = None
            cu_seqlens_k = cu_seqlens
        else:
            raise NotImplementedError("QUACK grouped GEMM epilogue supports only 2D/3D and 2D/2D grouped_mm")
        _, out = gemm_act(
            a,
            b,
            activation=None,
            store_preact=False,
            tuned=False,
            tensor_epilogue_fn=epilogue_fn,
            tensor_epilogue_key=epilogue_key,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            out_dtype=a.dtype if out_dtype is None else out_dtype,
        )
        return out
    if epilogue_arg_kinds and epilogue_arg_kinds != ("tile",):
        raise NotImplementedError(
            f"QUACK GEMM epilogue supports only one full-tile aux tensor for now, got {epilogue_arg_kinds}"
        )
    if epilogue_arg_kinds and len(epilogue_args) != 1:
        raise RuntimeError("full-tile QUACK epilogue requires exactly one epilogue arg")
    if epilogue_arg_kinds and C is not None:
        raise NotImplementedError("full-tile QUACK epilogue arg cannot be combined with C yet")
    if scale_a is not None or scale_b is not None:
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
    _, out = gemm_act(
        a,
        b,
        C=epilogue_args[0] if epilogue_arg_kinds else C,
        activation=None,
        store_preact=False,
        tuned=False,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_uses_c=bool(epilogue_arg_kinds),
        alpha=alpha,
        beta=beta,
    )
    return out


def gemm_relu(a: Tensor, b: Tensor) -> Tensor:
    _, out = gemm_act(a, b, activation="relu", store_preact=False, tuned=False)
    return out
