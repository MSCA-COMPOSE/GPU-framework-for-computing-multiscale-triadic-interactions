import argparse
import logging
import os
import socket

import numpy as np

import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, wait

from hpda_utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

if __name__ == '__main__':

    # Spanwise domain length used in the cell-volume calculation
    L_Z = 0.3

    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", help="DASK scheduler file")
    parser.add_argument("input_dir", help="Directory containing input raw files")
    parser.add_argument("sequence", help="Sequence file listing the snapshots to read")
    parser.add_argument("output_dir", help="Directory where the POD outputs will be written")
    parser.add_argument("nb_r", type=int, help="Number of snapshots to process")
    args = parser.parse_args()

    logging.info('Starting main')

    # Connect to the Dask scheduler created by the submission script
    client = Client(scheduler_file=args.scheduler)

    # Send the helper module to the workers
    client.upload_file('../hpda/hpda_utils.py')

    print(dask.config.config)

    # Print a forwarding hint for the Dask dashboard
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "login.cluster.address"
    current_user = os.environ.get('USER')
    logging.info(f"ssh -N -L {port}:{host}:{port} {current_user}@{login_node_address}")
    logging.info('Started')

    # Load snapshot sequence
    sequence_file = os.path.join(args.input_dir, args.sequence)
    seq = np.genfromtxt(sequence_file, dtype=int)
    snaps = args.nb_r

    logging.info(f"Number of snapshots: {snaps}")

    SUBSPACE_NAME = "FLOW_phys"
    blockid = 2
    chunk_size = 70000

    # Build the list of raw snapshot files to read
    files = []
    for num in seq[:snaps]:
        filename = f"{SUBSPACE_NAME}_{blockid}_{num}.raw"
        files.append(os.path.join(args.input_dir, filename))

    logging.info('Reading mesh')

    # Read the structured grid for the selected block
    gridname = f"{SUBSPACE_NAME}_GRID_{blockid}.xyz"
    nxp, nyp, nzp = read_grid_header(os.path.join(args.input_dir, gridname))
    totdim = nxp * nyp * nzp

    xyz = read_grid(os.path.join(args.input_dir, gridname), nxp, nyp, nzp)
    logging.info(xyz.shape)

    # Compute cell volumes once from the grid
    vol = compute_grid_volume(xyz, L_Z)
    logging.info(vol.shape)

    # Create delayed readers for all selected snapshots
    lazy_read = [read_file(filename, nxp, nyp, nzp) for filename in files]
    logging.info('Started Read')

    # Extract the three velocity components and reshape them as
    # (space, time) matrices for the POD
    Us = []
    Vs = []
    Ws = []

    for item in lazy_read:
        da_temp = da.from_delayed(item, dtype=np.float64, shape=(nxp, nyp, nzp, 5))
        Us.append(da_temp[..., 1].reshape(totdim, 1))
        Vs.append(da_temp[..., 2].reshape(totdim, 1))
        Ws.append(da_temp[..., 3].reshape(totdim, 1))

    u = da.concatenate(Us, axis=1).rechunk((chunk_size, -1))
    v = da.concatenate(Vs, axis=1).rechunk((chunk_size, -1))
    w = da.concatenate(Ws, axis=1).rechunk((chunk_size, -1))

    # The delayed readers are no longer needed after the matrices are built
    for item in lazy_read:
        del item

    # Remove the temporal mean at each spatial point
    u_mean = da.mean(u, axis=1, keepdims=True).rechunk((chunk_size, -1))
    v_mean = da.mean(v, axis=1, keepdims=True).rechunk((chunk_size, -1))
    w_mean = da.mean(w, axis=1, keepdims=True).rechunk((chunk_size, -1))

    u = u - u_mean
    v = v - v_mean
    w = w - w_mean

    logging.info('Writing mean fields')

    # Save the mean fields so they can be reused later by the triadic step
    dd.from_dask_array(
        u_mean, columns=['1']
    ).to_parquet(
        os.path.join(args.output_dir, f'mean_u_{blockid}.parquet'),
        engine='pyarrow'
    )

    dd.from_dask_array(
        v_mean, columns=['1']
    ).to_parquet(
        os.path.join(args.output_dir, f'mean_v_{blockid}.parquet'),
        engine='pyarrow'
    )

    dd.from_dask_array(
        w_mean, columns=['1']
    ).to_parquet(
        os.path.join(args.output_dir, f'mean_w_{blockid}.parquet'),
        engine='pyarrow'
    )

    # Assemble the full fluctuation matrix:
    # X = [u'; v'; w']
    X = da.concatenate([u, v, w])

    # Compute the singular value decomposition of the reduced test case
    svd_u, svd_s, svd_v = da.linalg.svd(X)

    # Only the singular values and right singular vectors are needed later
    del X, svd_u

    logging.info('Persisting SVD')
    svd_s, svd_v = dask.persist(svd_s, svd_v)
    wait([svd_s, svd_v])

    logging.info(svd_s.shape)
    logging.info(svd_v.shape)
    logging.info('SVD persist')

    # Use the snapshot identifiers as column labels for svd_v
    column_list = seq[:snaps].astype(str).tolist()

    dd.from_dask_array(
        svd_s.reshape(svd_s.size, 1),
        columns=['1']
    ).to_parquet(
        os.path.join(args.output_dir, f'svd_s_{snaps}.parquet'),
        engine='pyarrow'
    )

    dd.from_dask_array(
        svd_v,
        columns=column_list
    ).to_parquet(
        os.path.join(args.output_dir, f'svd_v_{snaps}.parquet'),
        engine='pyarrow'
    )

    logging.info('SVD Finished')
    logging.info('Completed')