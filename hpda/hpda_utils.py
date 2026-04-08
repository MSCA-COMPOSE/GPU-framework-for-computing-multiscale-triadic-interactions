import glob
import logging
import os

import dask
import dask.array as da
import numpy as np


def read_grid_header(fname):
    """Read the three grid dimensions stored at the beginning of the grid file."""
    return np.fromfile(fname, count=3, dtype='int32')


def read_grid(fname, nxp, nyp, nzp):
    """Read the structured grid coordinates from the binary grid file."""
    return np.fromfile(fname, offset=12, dtype='float32').reshape((nxp, nyp, nzp, 3), order='F')


@dask.delayed
def read_file(fname, nxp, nyp, nzp):
    """
    Read one raw flow snapshot.

    The file is assumed to contain five fields stored in Fortran order.
    Any NaNs are replaced with zeros to avoid propagating invalid values.
    """
    data = np.fromfile(fname, offset=28, dtype='float32').reshape((nxp, nyp, nzp, 5), order='F')
    data = np.nan_to_num(data, nan=0.0)
    return data


def polygon_area(x, y):
    """Compute the area of a quadrilateral from its corner coordinates."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def compute_grid_volume(grid, L_Z):
    """
    Compute approximate cell volumes for the structured grid.

    The cross-sectional cell area is evaluated on one spanwise plane and then
    extended uniformly along the span using the domain length L_Z.
    """
    nxp, nyp, nzp = grid.shape[:3]
    dz = L_Z / nzp

    x = np.zeros([nxp, nyp, nzp])
    y = np.zeros([nxp, nyp, nzp])
    volumes = np.zeros([nxp, nyp, nzp])

    x[:, :, 0] = grid[..., 0, 0]
    y[:, :, 0] = grid[..., 0, 1]

    for j in range(nyp - 1):
        for i in range(nxp - 1):
            xv = np.array([x[i, j, 0], x[i + 1, j, 0], x[i + 1, j + 1, 0], x[i, j + 1, 0]])
            yv = np.array([y[i, j, 0], y[i + 1, j, 0], y[i + 1, j + 1, 0], y[i, j + 1, 0]])
            volumes[i, j, 0] = dz * polygon_area(xv, yv)

    for k in range(nzp):
        volumes[:, :, k] = volumes[:, :, 0]

    return volumes


@dask.delayed
def read_npy_file(filename):
    """Read one temporary NumPy block written during the GPU triadic step."""
    return np.load(filename)


def load_and_sum_blocks_lazy(dump_dir, component, triple_svd=None):
    """
    Lazily read all temporary block files for one velocity component and sum them.

    If a precomputed temporal triple product is provided, it is applied at the end.
    """
    file_pattern = os.path.join(dump_dir, f"{component}_output_block_*.npy")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        raise FileNotFoundError(f"No .npy files found in {file_pattern}")

    sample_shape = np.load(file_list[0], mmap_mode="r").shape

    lazy_arrays = [
        da.from_delayed(read_npy_file(f), shape=sample_shape, dtype=np.float32)
        for f in file_list
    ]

    stacked = da.stack(lazy_arrays, axis=0)
    stacked = stacked.rechunk((1,) + sample_shape)
    summed = stacked.sum(axis=0)

    if triple_svd is None:
        return summed

    return triple_svd * summed


def log_variable_details(variable):
    """Log a few useful diagnostics for Dask arrays during debugging."""
    logging.info(f"Type of variable: {type(variable)}")

    if isinstance(variable, da.Array):
        logging.info(f"Shape of Dask array: {variable.shape}")
        logging.info(f"Data type (dtype) of Dask array: {variable.dtype}")
        logging.info(f"Number of partitions (chunks) in Dask array: {variable.npartitions}")
        logging.info(f"Chunk shape (chunk size) of Dask array: {variable.chunksize}")


def cleanup_component_dump(dump_dir, component):
    """Remove the temporary block files generated for one velocity component."""
    file_pattern = os.path.join(dump_dir, f"{component}_output_block_*.npy")

    for f in glob.glob(file_pattern):
        try:
            os.remove(f)
        except Exception as e:
            logging.warning(f"Failed to delete {f}: {e}")