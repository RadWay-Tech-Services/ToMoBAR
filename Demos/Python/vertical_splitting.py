import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import timeit
import tomophantom
from tomophantom import TomoP3D
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy


print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.3 * np.pi * N_size)  # angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)

print("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)
input_data_labels = ["detY", "angles", "detX"]

# transfering numpy array to CuPy array
projData3D_analyt_cupy = cp.asarray(projData3D_analyt, order="C")

# %%
block_sizes = [8, 16]
result_keys = [
    "final_tmp_p",
    "r2c_c1dfftshift_output",
    "fft_datac_output",
    "final_datac",
    "final_fde",
    "unpadded_recon_up",
    "recon_vol_block_size",
]
result = {}

for block_size in block_sizes:
    index_start = 0
    index_end = block_size
    for k in result_keys:
        if k == "final_tmp_p":
            result[f"{k}{block_size}"] = cp.empty(
                (
                    N_size,
                    projData3D_analyt_cupy.shape[1],
                    projData3D_analyt_cupy.shape[2],
                )
            )
        elif (
            k == "r2c_c1dfftshift_output"
            or k == "fft_datac_output"
            or k == "final_datac"
        ):
            result[f"{k}{block_size}"] = cp.empty(
                (
                    N_size // 2,
                    projData3D_analyt_cupy.shape[1],
                    projData3D_analyt_cupy.shape[2],
                ),
                dtype=cp.complex64,
            )
        elif k == "final_fde":
            result[f"{k}{block_size}"] = cp.empty(
                (
                    N_size // 2,
                    2 * projData3D_analyt_cupy.shape[2],
                    2 * projData3D_analyt_cupy.shape[2],
                ),
                dtype=cp.complex64,
            )
        elif k == "unpadded_recon_up":
            result[f"{k}{block_size}"] = cp.empty((N_size, N_size, N_size))
        elif k == "recon_vol_block_size":
            result[f"{k}{block_size}"] = cp.empty((N_size, N_size, N_size))

    for block_index in range(0, N_size // block_size):
        proj_data_block = projData3D_analyt_cupy[index_start:index_end, :, :]
        RecToolsCP = RecToolsDIRCuPy(
            DetectorsDimH=Horiz_det,  # Horizontal detector dimension
            DetectorsDimH_pad=0,  # Padding size of horizontal detector
            DetectorsDimV=proj_data_block.shape[
                0
            ],  # Vertical detector dimension (3D case)
            CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
            AnglesVec=angles_rad,  # A vector of projection angles in radians
            ObjSize=N_size,  # Reconstructed object dimensions (scalar)
            device_projector="gpu",
        )

        block_result = RecToolsCP.FOURIER_INV(
            proj_data_block,
            recon_mask_radius=0.95,
            data_axes_labels_order=input_data_labels,
        )
        for k in result_keys:
            if result[f"{k}{block_size}"].dtype == cp.complex64:
                result[f"{k}{block_size}"][
                    (block_index * (block_size // 2)) : (
                        (block_index + 1) * (block_size // 2)
                    ),
                    :,
                    :,
                ] = block_result[k]
            else:
                result[f"{k}{block_size}"][index_start:index_end, :, :] = block_result[
                    k
                ]
        index_start += block_size
        index_end += block_size

    for k in result_keys:
        result[f"{k}{block_size}"] = cp.asnumpy(result[f"{k}{block_size}"])

block_size_a = block_sizes[0]
block_size_b = block_sizes[1]

for k in result_keys:
    if result[f"{k}{block_size_a}"].dtype == cp.complex64:
        sliceSel = int(0.25 * N_size)
        complex_field_key = "imag"
        plt.figure()
        plt.subplot(331)
        plt.imshow(
            getattr(result[f"{k}{block_size_a}"][sliceSel, :, :], complex_field_key)
        )
        plt.colorbar()
        plt.title(f"block{block_size_a} {k}, axial view")
        plt.subplot(332)
        plt.imshow(
            getattr(result[f"{k}{block_size_a}"][:, sliceSel, :], complex_field_key)
        )
        plt.colorbar()
        plt.title(f"block{block_size_a} {k}, coronal view")
        plt.subplot(333)
        plt.imshow(
            getattr(result[f"{k}{block_size_a}"][:, :, sliceSel], complex_field_key)
        )
        plt.colorbar()
        plt.title(f"block{block_size_a} {k}, sagittal view")

        plt.subplot(334)
        plt.imshow(
            getattr(result[f"{k}{block_size_b}"][sliceSel, :, :], complex_field_key)
        )
        plt.colorbar()
        plt.title(f"block{block_size_b} {k}, axial view")
        plt.subplot(335)
        plt.imshow(
            getattr(result[f"{k}{block_size_b}"][:, sliceSel, :], complex_field_key)
        )
        plt.colorbar()
        plt.title(f"block{block_size_b} {k}, coronal view")
        plt.subplot(336)
        plt.imshow(
            getattr(result[f"{k}{block_size_b}"][:, :, sliceSel], complex_field_key)
        )
        plt.colorbar()
        plt.title(f"block{block_size_b} {k}, sagittal view")

        residual_im = getattr(
            result[f"{k}{block_size_a}"], complex_field_key
        ) - getattr(result[f"{k}{block_size_b}"], complex_field_key)

        plt.subplot(337)
        plt.imshow(residual_im[sliceSel, :, :])
        plt.colorbar()
        plt.title(f"diff {k}, axial view")
        plt.subplot(338)
        plt.imshow(residual_im[:, sliceSel, :])
        plt.colorbar()
        plt.title(f"diff {k}, coronal view")
        plt.subplot(339)
        plt.imshow(residual_im[:, :, sliceSel])
        plt.colorbar()
        plt.title(f"diff {k}, sagittal view")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()
    else:
        sliceSel = int(0.5 * N_size)
        plt.figure()
        plt.subplot(331)
        plt.imshow(result[f"{k}{block_size_a}"][sliceSel, :, :])
        plt.colorbar()
        plt.title(f"block{block_size_a} {k}, axial view")
        plt.subplot(332)
        plt.imshow(result[f"{k}{block_size_a}"][:, sliceSel, :])
        plt.colorbar()
        plt.title(f"block{block_size_a} {k}, coronal view")
        plt.subplot(333)
        plt.imshow(result[f"{k}{block_size_a}"][:, :, sliceSel])
        plt.colorbar()
        plt.title(f"block{block_size_a} {k}, sagittal view")

        plt.subplot(334)
        plt.imshow(result[f"{k}{block_size_b}"][sliceSel, :, :])
        plt.colorbar()
        plt.title(f"block{block_size_b} {k}, axial view")
        plt.subplot(335)
        plt.imshow(result[f"{k}{block_size_b}"][:, sliceSel, :])
        plt.colorbar()
        plt.title(f"block{block_size_b} {k}, coronal view")
        plt.subplot(336)
        plt.imshow(result[f"{k}{block_size_b}"][:, :, sliceSel])
        plt.colorbar()
        plt.title(f"block{block_size_b} {k}, sagittal view")

        residual_im = result[f"{k}{block_size_a}"] - result[f"{k}{block_size_b}"]

        plt.subplot(337)
        plt.imshow(residual_im[sliceSel, :, :])
        plt.colorbar()
        plt.title(f"diff {k}, axial view")
        plt.subplot(338)
        plt.imshow(residual_im[:, sliceSel, :])
        plt.colorbar()
        plt.title(f"diff {k}, coronal view")
        plt.subplot(339)
        plt.imshow(residual_im[:, :, sliceSel])
        plt.colorbar()
        plt.title(f"diff {k}, sagittal view")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()

        res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
        print(res_norm)
