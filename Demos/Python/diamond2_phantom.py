#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pathlib
import numpy as np
import tomophantom

import h5py
from httomo.methods import save_intermediate_data
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock


def save_nxs(
    file: h5py.File,
    aux_data: AuxiliaryData,
    data,
    chunk_start,
    global_shape,
    detector_x,
    detector_y,
):
    b1 = DataSetBlock(
        data=data,
        aux_data=aux_data,
        slicing_dim=1,
        block_start=0,
        chunk_start=chunk_start,
        chunk_shape=data.shape,
        global_shape=global_shape,
    )

    save_intermediate_data(
        b1.data,
        b1.global_shape,
        b1.global_index,
        b1.slicing_dim,
        file,
        frames_per_chunk=0,
        minimum_block_length=global_shape[0],
        path="/data",
        detector_x=detector_x,
        detector_y=detector_y,
        angles=b1.angles,
    )


def main(args: argparse.Namespace):
    print("Building 3D phantom using TomoPhantom software")
    model = args.model_number
    path = pathlib.Path(tomophantom.__file__).parent
    path_library3D = str(path / "phantomlib" / "Phantom3DLibrary.dat")

    N_size = args.phantom_size
    chunk_size = 10
    chunk_count = int(np.ceil(N_size / chunk_size))

    # Projection geometry related parameters:
    Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
    Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
    angles_num = int(0.3 * np.pi * N_size)  # angles number
    global_shape = (angles_num, Vert_det, Horiz_det)

    angles = np.linspace(*args.angle_range, angles_num, dtype="float32")  # in degrees
    aux_data = AuxiliaryData(angles=angles)

    with h5py.File(args.output_path, "w") as file:
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

    for i in range(chunk_count):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, N_size)

        projData3D_analyt = tomophantom.TomoP3D.ModelSinoSub(
            model,
            N_size,
            Horiz_det,
            Vert_det,
            (chunk_start, chunk_end),
            angles,
            path_library3D,
        )
        swapped_projData3D_analyt = np.swapaxes(projData3D_analyt, 0, 1)

        with h5py.File(args.output_path, "a") as file:
            save_nxs(
                file,
                aux_data,
                swapped_projData3D_analyt,
                chunk_start,
                global_shape,
                Horiz_det,
                Vert_det,
            )

    print(
        f"#slices: {Vert_det}, #projections: {angles_num}, detector width: {Horiz_det}"
    )
    print(f"phantom shape in output file: {global_shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TomoPhantom phantom .nxs generator. Output can be used with httomo, e.g. 'httomo run phantom.nxs pipeline.yaml output'.")
    parser.add_argument(
        "-m",
        "--model-number",
        type=int,
        default=13,  # Shepp-Logan
        help="Model number in TomoPhantom's model library. See https://github.com/dkazanc/TomoPhantom/tree/master/tomophantom/phantomlib.",
    )
    parser.add_argument(
        "-s",
        "--phantom-size",
        type=int,
        default=256,
        # 2048  # Define phantom dimensions using a scalar value (cubic phantom) -- Limit for machines with 32 GB RAM
        # 4096  # Define phantom dimensions using a scalar value (cubic phantom) -- Diamond2 resolution
        # The biggest camera so far (that we are still commissioning) is 4416 x 2368, and if we use it in 2FOV mode that will be a uint16 4416 x 2368 x 3600.
        # 4416 is the horizontal, 2368 is the vertical detector dimension.
        help="Cubic phantom size.",
    )
    parser.add_argument(
        "-a",
        "--angle-range",
        nargs=2,
        metavar=("start", "end"),
        type=float,
        default=(0.0, 179.9),
    )
    parser.add_argument(
        "-o", "--output-path", type=pathlib.Path, default="./phantom.nxs"
    )

    args = parser.parse_args()
    main(args)
