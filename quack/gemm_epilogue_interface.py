from collections.abc import Callable

from torch import Tensor

from quack.gemm_interface import gemm_act


def gemm_epilogue(
    a: Tensor,
    b: Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    C: Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    _, out = gemm_act(
        a,
        b,
        C=C,
        activation=None,
        store_preact=False,
        tuned=False,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
    )
    if alpha != 1.0 or beta != 1.0:
        raise NotImplementedError("gemm_epilogue scalar alpha/beta is not wired yet")
    return out


def gemm_relu(a: Tensor, b: Tensor) -> Tensor:
    _, out = gemm_act(a, b, activation="relu", store_preact=False, tuned=False)
    return out
