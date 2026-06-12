import cutlass
import pytest
from cutlass._mlir import ir

from quack.epi_ops import ColVecReduce, EpiSmemBytes, RowVecLoad, TileLoad, TileStore


class _ArgTensor:
    element_type = cutlass.Float32


@pytest.fixture(scope="module")
def mlir_context():
    with ir.Context():
        yield


def test_colvec_reduce_smem_bytes_default_to_direct_write_path(mlir_context):
    op = ColVecReduce("mColVecReduce")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes()


def test_colvec_reduce_smem_bytes_use_atom_layout_n_warp_staging(mlir_context):
    op = ColVecReduce("mColVecReduce")

    # op.smem_bytes is only called for active (non-None) args by the framework;
    # ComposableEpiMixin.epi_smem_bytes filters from `args` before calling.
    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32), (4, 2, 1)) == EpiSmemBytes(
        unstaged=64 * 1 * 4
    )


def test_vecload_smem_bytes_are_unstaged(mlir_context):
    op = RowVecLoad("mRowVecBroadcast")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
        unstaged=128 * 4
    )


def test_tile_store_smem_bytes_are_d_staged(mlir_context):
    op = TileStore("mAuxOut")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
        d_stage=64 * 32 * 4
    )


def test_tile_load_smem_accounting_is_c_stage(mlir_context):
    op = TileLoad("mTile")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
        c_stage=64 * 32 * 4
    )
