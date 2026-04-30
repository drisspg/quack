from collections.abc import Callable

from torch import Tensor

from quack.gemm_interface import gemm_act


def gemm_epilogue(a: Tensor, b: Tensor, epilogue_fn: Callable, epilogue_key: str) -> Tensor:
    _, out = gemm_act(
        a,
        b,
        activation=None,
        store_preact=False,
        tuned=False,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
    )
    return out


def gemm_relu(a: Tensor, b: Tensor) -> Tensor:
    _, out = gemm_act(a, b, activation="relu", store_preact=False, tuned=False)
    return out
