#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# httomo run diamond2_phantom.nxs ~/work/httomo_pipeline/diamond2_phantom.yaml ~/work/output
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

import h5py
from pathlib import Path
from httomo.methods import save_intermediate_data
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock

plot = False
# plot = True

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
# N_size = 2048  # Define phantom dimensions using a scalar value (cubic phantom) -- Limit for machines with 32 GB RAM
# N_size = 4096  # Define phantom dimensions using a scalar value (cubic phantom) -- Diamond2 resolution
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

sliceSel = int(0.5 * N_size)
max_val = 1

print(f"phantom shape: {phantom_tm.shape}")
if plot:
    plt.figure()
    plt.subplot(131)
    plt.imshow(phantom_tm[sliceSel, :, :])
    plt.colorbar()
    plt.title("phantom, axial view")

    plt.subplot(132)
    plt.imshow(phantom_tm[:, sliceSel, :])
    plt.colorbar()
    plt.title("phantom, coronal view")

    plt.subplot(133)
    plt.imshow(phantom_tm[:, :, sliceSel])
    plt.colorbar()
    plt.title("phantom, sagittal view")
    plt.show()


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

print(f"#slices: {Vert_det}, #projections: {angles_num}, detector width: {Horiz_det}")
print(f"sinogram shape: {projData3D_analyt.shape}")
if plot:
    plt.figure()
    plt.subplot(131)
    plt.imshow(projData3D_analyt[int(0.5 * Vert_det), :, :])
    plt.title("sinogram, axial view")

    plt.subplot(132)
    plt.imshow(projData3D_analyt[:, int(0.5 * angles_num), :])
    plt.title("sinogram, coronal view")

    plt.subplot(133)
    plt.imshow(projData3D_analyt[:, :, int(0.5 * Horiz_det)])
    plt.title("sinogram, sagittal view")
    plt.show()


def save_nxs(filepath: Path, angles, data, detector_x, detector_y):
    # use increasing numbers in the data, to make sure blocks have different content
    aux_data = AuxiliaryData(angles=angles)
    # bsize = 3
    b1 = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=0,
        block_start=0,
        chunk_start=0,
        chunk_shape=data.shape,
        global_shape=data.shape,
    )
    # b1 = DataSetBlock(
    #     data=data[:bsize],
    #     aux_data=aux_data,
    #     slicing_dim=0,
    #     block_start=0,
    #     chunk_start=0,
    #     chunk_shape=GLOBAL_SHAPE,
    #     global_shape=GLOBAL_SHAPE,
    # )
    # b2 = DataSetBlock(
    #     data=global_data[bsize:],
    #     aux_data=aux_data,
    #     slicing_dim=0,
    #     block_start=bsize,
    #     chunk_start=0,
    #     chunk_shape=GLOBAL_SHAPE,
    #     global_shape=GLOBAL_SHAPE,
    # )

    with h5py.File(filepath, "w") as file:
        file.attrs["default"] = "entry"
        entry = file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        entry.attrs["NX_application"] = "NXtomo"

        entry.create_dataset("definition", data="NXtomo") 
        data_group = entry.create_group("data")
        data_group["data"] = h5py.SoftLink("/data") 
        data_group["rotation_angle"] = h5py.SoftLink("/angles") 

        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector")
        image_key_data = np.zeros_like(angles, dtype=np.int8)
        detector.create_dataset("image_key", data=image_key_data, dtype=np.int8)

        # # save in 2 blocks, starting with the second to confirm order-independence
        # save_intermediate_data(
        #     b2.data,
        #     b2.global_shape,
        #     b2.global_index,
        #     b2.slicing_dim,
        #     file,
        #     frames_per_chunk=0,
        #     minimum_block_length=GLOBAL_SHAPE[0],
        #     path="/data",
        #     detector_x=10,
        #     detector_y=20,
        #     angles=b2.angles,
        # )
        save_intermediate_data(
            b1.data,
            b1.global_shape,
            b1.global_index,
            b1.slicing_dim,
            file,
            frames_per_chunk=0,
            minimum_block_length=data.shape[0],
            path="/data",
            detector_x=detector_x,
            detector_y=detector_y,
            angles=b1.angles,
        )

save_nxs("./diamond2_phantom.nxs", angles, projData3D_analyt, Horiz_det, Vert_det)


# input_data_labels = ["detY", "angles", "detX"]

# # transfering numpy array to CuPy array
# projData3D_analyt_cupy = cp.asarray(projData3D_analyt, order="C")
# # %%
# # It is recommend to re-run twice in order to get the optimal time
# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print("%%%%%%%%%Reconstructing with 3D FBP-CuPy method %%%%%%%%%%%%")
# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# RecToolsCP = RecToolsDIRCuPy(
#     DetectorsDimH=Horiz_det,  # Horizontal detector dimension
#     DetectorsDimH_pad=0,  # Padding size of horizontal detector
#     DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
#     CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
#     AnglesVec=angles_rad,  # A vector of projection angles in radians
#     ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#     device_projector="gpu",
# )

# tic = timeit.default_timer()
# FBPrec_cupy = RecToolsCP.FBP(
#     projData3D_analyt_cupy,
#     recon_mask_radius=0.95,
#     data_axes_labels_order=input_data_labels,
#     cutoff_freq=0.3,
# )
# toc = timeit.default_timer()
# Run_time = toc - tic
# print(
#     "FBP 3D reconstruction with FFT filtering using CuPy (GPU) in {} seconds".format(
#         Run_time
#     )
# )

# # bring data from the device to the host
# FBPrec_cupy = cp.asnumpy(FBPrec_cupy)

# sliceSel = int(0.5 * N_size)
# max_val = 1
# plt.figure()
# plt.subplot(131)
# plt.imshow(FBPrec_cupy[sliceSel, :, :], vmin=0, vmax=max_val)
# plt.title("3D FBP Reconstruction, axial view")

# plt.subplot(132)
# plt.imshow(FBPrec_cupy[:, sliceSel, :], vmin=0, vmax=max_val)
# plt.title("3D FBP Reconstruction, coronal view")

# plt.subplot(133)
# plt.imshow(FBPrec_cupy[:, :, sliceSel], vmin=0, vmax=max_val)
# plt.title("3D FBP Reconstruction, sagittal view")
# plt.show()


# #
# sliceSel = int(0.5 * N_size)
# max_val = 0.3
# plt.figure()
# plt.subplot(131)
# plt.imshow(
#     abs(FBPrec_cupy[sliceSel, :, :] - phantom_tm[sliceSel, :, :]), vmin=0, vmax=max_val
# )
# plt.title("3D FBP residual, axial view")

# plt.subplot(132)
# plt.imshow(
#     abs(FBPrec_cupy[:, sliceSel, :] - phantom_tm[:, sliceSel, :]), vmin=0, vmax=max_val
# )
# plt.title("3D FBP residual, coronal view")

# plt.subplot(133)
# plt.imshow(
#     abs(FBPrec_cupy[:, :, sliceSel] - phantom_tm[:, :, sliceSel]), vmin=0, vmax=max_val
# )
# plt.title("3D FBP residual, sagittal view")
# plt.show()


# print(
#     "Min {} and Max {} of the volume".format(np.min(FBPrec_cupy), np.max(FBPrec_cupy))
# )

# # calculate errors
# Qtools = QualityTools(phantom_tm, FBPrec_cupy)
# RMSE = Qtools.rmse()
# print("Root Mean Square Error is {} for FBP".format(RMSE))

# # %%
# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print("%%%%%%%%%Reconstructing with 3D Fourier-CuPy method %%%%%%%%")
# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# RecToolsCP = RecToolsDIRCuPy(
#     DetectorsDimH=Horiz_det,  # Horizontal detector dimension
#     DetectorsDimH_pad=0,  # Padding size of horizontal detector
#     DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
#     CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
#     AnglesVec=angles_rad,  # A vector of projection angles in radians
#     ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#     device_projector="gpu",
# )

# tic = timeit.default_timer()
# Fourier_cupy = RecToolsCP.FOURIER_INV(
#     projData3D_analyt_cupy,
#     recon_mask_radius=0.95,
#     data_axes_labels_order=input_data_labels,
# )
# toc = timeit.default_timer()
# Run_time = toc - tic
# print("Fourier 3D reconstruction using CuPy (GPU) in {} seconds".format(Run_time))

# # bring data from the device to the host
# Fourier_cupy = cp.asnumpy(Fourier_cupy)

# sliceSel = int(0.5 * N_size)
# max_val = 1
# plt.figure()
# plt.subplot(131)
# plt.imshow(Fourier_cupy[sliceSel, :, :], vmin=0, vmax=max_val)
# plt.title("3D Fourier Reconstruction, axial view")

# plt.subplot(132)
# plt.imshow(Fourier_cupy[:, sliceSel, :], vmin=0, vmax=max_val)
# plt.title("3D Fourier Reconstruction, coronal view")

# plt.subplot(133)
# plt.imshow(Fourier_cupy[:, :, sliceSel], vmin=0, vmax=max_val)
# plt.title("3D Fourier Reconstruction, sagittal view")
# plt.show()


# sliceSel = int(0.5 * N_size)
# max_val = 0.3
# plt.figure()
# plt.subplot(131)
# plt.imshow(
#     abs(Fourier_cupy[sliceSel, :, :] - phantom_tm[sliceSel, :, :]), vmin=0, vmax=max_val
# )
# plt.title("3D Fourier residual, axial view")

# plt.subplot(132)
# plt.imshow(
#     abs(Fourier_cupy[:, sliceSel, :] - phantom_tm[:, sliceSel, :]), vmin=0, vmax=max_val
# )
# plt.title("3D Fourier residual, coronal view")

# plt.subplot(133)
# plt.imshow(
#     abs(Fourier_cupy[:, :, sliceSel] - phantom_tm[:, :, sliceSel]), vmin=0, vmax=max_val
# )
# plt.title("3D Fourier residual, sagittal view")
# plt.show()


# print(
#     "Min {} and Max {} of the volume".format(np.min(FBPrec_cupy), np.max(FBPrec_cupy))
# )

# # calculate errors
# Qtools = QualityTools(phantom_tm, Fourier_cupy)
# RMSE = Qtools.rmse()
# print("Root Mean Square Error is {} for Fourier inversion".format(RMSE))
