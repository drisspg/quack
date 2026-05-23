import torch

from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    _force_local_reduce_config,
    _is_safe_local_reduce_config,
    gemm_act_tuned,
    prune_invalid_gemm_configs,
)


def test_gemm_act_tuned_key_includes_epilogue_local_reduce_compile_knobs():
    required = {
        "tensor_epilogue_key",
        "tensor_epilogue_uses_c",
        "tensor_epilogue_returns_aux",
        "tensor_epilogue_arg_kinds",
        "tensor_epilogue_rowvec_biases",
        "tensor_epilogue_colvec_biases",
        "tensor_epilogue_tile_biases",
        "local_reduce_group",
        "local_reduce_dim",
        "local_reduce_op",
        "local_reduce_scale",
        "local_reduce_max_power",
        "local_reduce_feeds_main",
        "local_reduce_source_from_epilogue",
        "main_output_transform_group",
        "concat_layout",
    }

    assert required <= set(gemm_act_tuned.keys)


def test_force_local_reduce_config_returns_safe_family():
    unsafe = GemmConfig(tile_m=256, tile_n=256, cluster_m=2, cluster_n=2, device_capacity=10)

    n_reduce = _force_local_reduce_config(unsafe, group=32, dim=1)
    m_reduce = _force_local_reduce_config(unsafe, group=16, dim=0)

    assert _is_safe_local_reduce_config(n_reduce)
    assert _is_safe_local_reduce_config(m_reduce)
    assert n_reduce.tile_n == 128
    assert m_reduce.tile_n == 128


def test_prune_invalid_gemm_configs_keeps_only_safe_local_reduce_family(monkeypatch):
    import quack.gemm_interface as gi

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (10, 0))
    configs = [
        AutotuneConfig(
            config=GemmConfig(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, device_capacity=10)
        ),
        AutotuneConfig(
            config=GemmConfig(tile_m=128, tile_n=256, cluster_m=1, cluster_n=1, device_capacity=10)
        ),
        AutotuneConfig(
            config=GemmConfig(tile_m=256, tile_n=128, cluster_m=2, cluster_n=1, device_capacity=10)
        ),
    ]

    pruned = prune_invalid_gemm_configs(
        configs,
        {},
        A=torch.empty((128, 64)),
        B=torch.empty((64, 128)),
        local_reduce_feeds_main=True,
        local_reduce_group=32,
        local_reduce_dim=1,
    )

    assert [conf.kwargs["config"].tile_n for conf in pruned] == [128]
    assert all(_is_safe_local_reduce_config(conf.kwargs["config"]) for conf in pruned)
