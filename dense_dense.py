import ctypes
import ctypes.util
import pathlib

import numpy as np

from mlir.ir import Context, Module
import mlir.runtime as rt
from mlir import execution_engine, passmanager


MLIR_C_RUNNER_UTILS = ctypes.util.find_library("mlir_c_runner_utils")


with Context():
    module_add = Module.parse(
    """
    #Dense = #sparse_tensor.encoding<{
        map = (i, j) -> (i : dense, j : dense), posWidth = 64, crdWidth = 64
    }>

    #map = affine_map<(d0, d1) -> (d0, d1)>
    func.func @add(%st_0 : tensor<3x4xf64, #Dense>, %st_1 : tensor<3x4xf64, #Dense>) attributes { llvm.emit_c_interface } {
        %out_st = tensor.empty() : tensor<3x4xf64, #Dense>
        %res = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%st_0, %st_1 : tensor<3x4xf64, #Dense>, tensor<3x4xf64, #Dense>) outs(%out_st : tensor<3x4xf64, #Dense>) {
            ^bb0(%in_0: f64, %in_1: f64, %out: f64):
                %2 = sparse_tensor.binary %in_0, %in_1 : f64, f64 to f64
                    overlap = {
                        ^bb0(%arg1: f64, %arg2: f64):
                            %3 = arith.addf %arg1, %arg2 : f64
                            sparse_tensor.yield %3 : f64
                    }
                    left = {
                        ^bb0(%arg1: f64):
                            sparse_tensor.yield %arg1 : f64
                    }
                    right = {
                        ^bb0(%arg1: f64):
                            sparse_tensor.yield %arg1 : f64
                    }
                linalg.yield %2 : f64
        } -> tensor<3x4xf64, #Dense>

        sparse_tensor.print %res : tensor<3x4xf64, #Dense>

        return
    }
    """
    )

    CWD = pathlib.Path(".")
    (CWD / "module.mlir").write_text(str(module_add))

    pm = passmanager.PassManager.parse("builtin.module(sparse-assembler{direct-out=true}, sparsifier{create-sparse-deallocs=1 enable-runtime-library=false})")
    pm.run(module_add.operation)

    (CWD / "module_opt.mlir").write_text(str(module_add))

    ee_add = execution_engine.ExecutionEngine(module_add, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])

    dense_arr = np.array([[1, 0, 0, 1], [0, 2, 2, 0], [0, 0, 0, 1]], dtype=np.float64)
    dense_arr_memref = rt.get_ranked_memref_descriptor(dense_arr.ravel())

    dense_arr_2 = np.array([[1, 0, 0, 1], [0, 2, 2, 0], [0, 0, 0, 1]], dtype=np.float64)
    dense_arr_memref_2 = rt.get_ranked_memref_descriptor(dense_arr_2.ravel())

    ret = ee_add.invoke(
        "add",
        # Dense input
        ctypes.pointer(ctypes.pointer(dense_arr_memref)),
        # Dense input
        ctypes.pointer(ctypes.pointer(dense_arr_memref_2)),
    )
