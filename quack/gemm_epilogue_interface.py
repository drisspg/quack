from collections.abc import Callable

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
) -> Tensor:
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
        C=C,
        activation=None,
        store_preact=False,
        tuned=False,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
        alpha=alpha,
        beta=beta,
    )
    return out


def gemm_relu(a: Tensor, b: Tensor) -> Tensor:
    _, out = gemm_act(a, b, activation="relu", store_preact=False, tuned=False)
    return out
