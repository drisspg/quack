from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from typing import Any

import torch
from torch import Tensor


TENSOR_META_TAG = "__quack_tensor_meta__"
EPILOGUE_SOURCE_TAG = "__quack_epilogue_from_source__"

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
    "torch.float8_e8m0fnu": torch.float8_e8m0fnu,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def serialize_worker_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            TENSOR_META_TAG: True,
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return tuple(serialize_worker_value(v) for v in value)
    if isinstance(value, list):
        return [serialize_worker_value(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_worker_value(v) for k, v in value.items()}
    return value


def deserialize_worker_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get(TENSOR_META_TAG):
        return torch.empty_strided(
            value["shape"], value["stride"], dtype=_DTYPE_MAP[value["dtype"]], device="cuda"
        )
    if isinstance(value, tuple):
        return tuple(deserialize_worker_value(v) for v in value)
    if isinstance(value, list):
        return [deserialize_worker_value(v) for v in value]
    if isinstance(value, dict):
        return {k: deserialize_worker_value(v) for k, v in value.items()}
    return value


def make_epilogue_source_marker(epilogue_key: str | None, source: str) -> dict[str, Any]:
    return {
        EPILOGUE_SOURCE_TAG: True,
        "name": epilogue_key,
        "source": source,
    }


def is_epilogue_source_marker(value: Any) -> bool:
    return isinstance(value, dict) and value.get(EPILOGUE_SOURCE_TAG)


def load_epilogue_from_source(marker: dict[str, Any]) -> Any:
    source = marker["source"]
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    module_name = f"quack_generated_epilogue_{digest}"
    module_dir = os.path.join(tempfile.gettempdir(), "quack_generated_epilogues")
    os.makedirs(module_dir, exist_ok=True)
    module_path = os.path.join(module_dir, f"{module_name}.py")
    if not os.path.exists(module_path):
        with open(module_path, "w") as f:
            f.write(source)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated epilogue module {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    epilogue_fn = getattr(mod, marker["name"])
    setattr(epilogue_fn, "__quack_cache_key__", f"epilogue:{marker['name']}")
    return epilogue_fn
