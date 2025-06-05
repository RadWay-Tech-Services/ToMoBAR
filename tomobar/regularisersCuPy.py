"""Adding CuPy-enabled regularisers from the CCPi-regularisation toolkit and
instantiate a proximal operator for iterative methods.
"""

import cupy as cp
from typing import Optional
from tomobar.cuda_kernels import load_cuda_module

try:
    from ccpi.filters.regularisersCuPy import ROF_TV as ROF_TV_cupy
except ImportError:
    print(
        "____! CCPi-regularisation package (CuPy part needed only) is missing, please install !____"
    )


def prox_regul(self, X: cp.ndarray, _regularisation_: dict) -> cp.ndarray:
    """Enabling proximal operators step in iterative reconstruction.

    Args:
        X (cp.ndarray): 2D or 3D CuPy array.
        _regularisation_ (dict): Regularisation dictionary with parameters, see :mod:`tomobar.supp.dicts`.

    Returns:
        cp.ndarray: Filtered 2D or 3D CuPy array.
    """
    info_vec = (_regularisation_["iterations"], 0)
    # The proximal operator of the chosen regulariser
    if "ROF_TV" in _regularisation_["method"]:
        # Rudin - Osher - Fatemi Total variation method
        X_prox = ROF_TV_cupy(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            self.Atools.device_index,
        )
    if "PD_TV" in _regularisation_["method"]:
        # Primal-Dual (PD) Total variation method by Chambolle-Pock
        X_prox_orig = PD_TV_cupy(
            X,
            True,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.Atools.device_index,
        )
        # X_prox = X_prox_orig
        X_prox = PD_TV_cupy(
            X,
            False,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.Atools.device_index,
        )

        if True: 
            diff = (X_prox - X_prox_orig).get()
            shape = diff.shape
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(131)
            plt.imshow(diff[shape[0] // 2, :, :])
            plt.colorbar()
            plt.title("diff, axial view")

            plt.subplot(132)
            plt.imshow(diff[:, shape[1] // 2, :])
            plt.colorbar()
            plt.title("diff, coronal view")

            plt.subplot(133)
            plt.imshow(diff[:, :, shape[2] // 2])
            plt.colorbar()
            plt.title("diff, sagittal view")
            plt.show()
        else:
            X_prox = X_prox_orig

    return X_prox


def PD_TV_cupy(
    data: cp.ndarray,
    use_original_code: bool,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 1000,
    methodTV: Optional[int] = 0,
    nonneg: Optional[int] = 0,
    lipschitz_const: Optional[float] = 8.0,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """Primal Dual algorithm for non-smooth convex Total Variation functional.
       Ref: Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
       with Applications to Imaging", 2010.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (Optional[float], optional): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (Optional[int], optional): The number of iterations. Defaults to 1000.
        methodTV (Optional[int], optional): Choose between isotropic (0) or anisotropic (1) case for TV norm.
        nonneg (Optional[int], optional): Enable non-negativity in updates by selecting 1. Defaults to 0.
        lipschitz_const (Optional[float], optional): Lipschitz constant to control convergence.
        gpu_id (Optional[int], optional): A GPU device index to perform operation on. Defaults to 0.

    Returns:
        cp.ndarray: PD-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")

    # with cp.cuda.Device(gpu_id):
    cp.get_default_memory_pool().free_all_blocks()

    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    # prepare some parameters:
    tau = cp.float32(regularisation_parameter * 0.1)
    sigma = cp.float32(1.0 / (lipschitz_const * tau))
    theta = cp.float32(1.0)
    lt = cp.float32(tau / regularisation_parameter)

    # initialise CuPy arrays here:
    out = data.copy()
    P1 = cp.zeros(data.shape, dtype=cp.float32, order="C")
    P2 = cp.zeros(data.shape, dtype=cp.float32, order="C")

    if use_original_code:
        d_old = cp.empty(data.shape, dtype=cp.float32, order="C")
    else:
        U_arrays = [out, cp.zeros(data.shape, dtype=cp.float32)]

    # loading and compiling CUDA kernels:
    if data.ndim == 3:
        data3d = True
        P3 = cp.zeros(data.shape, dtype=cp.float32, order="C")
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)
        module = load_cuda_module("primal_dual_for_total_variation")
        primal_dual_for_total_variation = module.get_function(
            "primal_dual_for_total_variation_3D"
        )
        dualPD_kernel = module.get_function("dualPD3D_kernel")
        Proj_funcPD_iso_kernel = module.get_function("Proj_funcPD3D_iso_kernel")
        Proj_funcPD_aniso_kernel = module.get_function("Proj_funcPD3D_aniso_kernel")
        DivProj_kernel = module.get_function("DivProj3D_kernel")
        PDnonneg_kernel = module.get_function("PDnonneg3D_kernel")
        getU_kernel = module.get_function("getU3D_kernel")
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        module = load_cuda_module("primal_dual_for_total_variation")
        primal_dual_for_total_variation = module.get_function(
            "primal_dual_for_total_variation_2D"
        )
        dualPD_kernel = module.get_function("dualPD_kernel")
        Proj_funcPD_iso_kernel = module.get_function("Proj_funcPD2D_iso_kernel")
        Proj_funcPD_aniso_kernel = module.get_function("Proj_funcPD2D_aniso_kernel")
        DivProj_kernel = module.get_function("DivProj2D_kernel")
        PDnonneg_kernel = module.get_function("PDnonneg2D_kernel")
        getU_kernel = module.get_function("getU2D_kernel")

    # perform algorithm iterations
    for iter in range(iterations):
        if use_original_code:
            # calculate differences
            if data3d:
                params1 = (out, P1, P2, P3, sigma, dx, dy, dz)
            else:
                params1 = (out, P1, P2, sigma, dx, dy)
            dualPD_kernel(
                grid_dims, block_dims, params1
            )  # computing the the dual P variable
            cp.cuda.runtime.deviceSynchronize()

            if nonneg != 0:
                if data3d:
                    params2 = (out, dx, dy, dz)
                else:
                    params2 = (out, dx, dy)
                PDnonneg_kernel(grid_dims, block_dims, params2)
                cp.cuda.runtime.deviceSynchronize()

            if data3d:
                params3 = (P1, P2, P3, dx, dy, dz)
            else:
                params3 = (P1, P2, dx, dy)
            if methodTV == 0:
                Proj_funcPD_iso_kernel(grid_dims, block_dims, params3)
            else:
                Proj_funcPD_aniso_kernel(grid_dims, block_dims, params3)
            cp.cuda.runtime.deviceSynchronize()

            d_old = out.copy()

            if data3d:
                params4 = (out, data, P1, P2, P3, lt, tau, dx, dy, dz)
            else:
                params4 = (out, data, P1, P2, lt, tau, dx, dy)
            DivProj_kernel(grid_dims, block_dims, params4)  # calculate divergence
            cp.cuda.runtime.deviceSynchronize()

            if data3d:
                params5 = (out, d_old, theta, dx, dy, dz)
            else:
                params5 = (out, d_old, theta, dx, dy)
            getU_kernel(grid_dims, block_dims, params5)
            cp.cuda.runtime.deviceSynchronize()
        else:
            if data3d:
                print(f"iter: {iter}, iter % 2: {iter % 2}, (iter + 1) % 2: {(iter + 1) % 2}")
                params = (
                    data,
                    U_arrays[iter % 2],
                    U_arrays[(iter + 1) % 2],
                    P1,
                    P2,
                    P3,
                    sigma,
                    tau,
                    lt,
                    theta,
                    dx,
                    dy,
                    dz,
                    nonneg,
                    methodTV,
                )
            else:
                params = (out, d_old, theta, dx, dy)
            primal_dual_for_total_variation(grid_dims, block_dims, params)

    if not use_original_code:
        out = U_arrays[iterations % 2]
    return out
