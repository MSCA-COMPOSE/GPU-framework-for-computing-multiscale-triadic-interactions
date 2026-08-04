import argparse
import logging
import os
import socket

import numpy as np

import dask
import dask.array as da
from dask.distributed import Client, wait

from hpda_utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

if __name__ == '__main__':

    # Block-SPOD settings of the reduced test case:
    # nsnaps = nblockf x nDFT non-overlapping blocks, rectangular window,
    # two-sided FFT, no folding (same conventions as the production case).
    nblockf = 5
    nDFT = 6

    # Spatial chunk of the (space, time) matrices
    spod_chunk = 70000

    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", help="DASK scheduler file")
    parser.add_argument("input_dir", help="Directory containing input raw files")
    parser.add_argument("sequence", help="Sequence file listing the snapshots to read")
    parser.add_argument("output_dir", help="Directory where the SPOD outputs will be written")
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
    logging.info(f"SPOD nblockf = {nblockf}, nDFT = {nDFT}")

    SUBSPACE_NAME = "FLOW_phys"
    blockid = 2

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

    # Create delayed readers for all selected snapshots
    lazy_read = [read_file(filename, nxp, nyp, nzp) for filename in files]
    logging.info('Started Read')

    # Extract the three velocity components and reshape them as
    # (space, time) matrices, one full time record per spatial chunk
    Us = []
    Vs = []
    Ws = []

    for item in lazy_read:
        da_temp = da.from_delayed(item, dtype=np.float32, shape=(nxp, nyp, nzp, 5))
        Us.append(da_temp[..., 1].reshape(totdim, 1))
        Vs.append(da_temp[..., 2].reshape(totdim, 1))
        Ws.append(da_temp[..., 3].reshape(totdim, 1))

    u = da.concatenate(Us, axis=1).rechunk((spod_chunk, snaps))
    v = da.concatenate(Vs, axis=1).rechunk((spod_chunk, snaps))
    w = da.concatenate(Ws, axis=1).rechunk((spod_chunk, snaps))

    # The delayed readers are no longer needed after the matrices are built
    for item in lazy_read:
        del item

    # Remove the temporal mean at each spatial point
    u = (u - da.mean(u, axis=1, keepdims=True)).astype(np.float32)
    v = (v - da.mean(v, axis=1, keepdims=True)).astype(np.float32)
    w = (w - da.mean(w, axis=1, keepdims=True)).astype(np.float32)

    u, v, w = dask.persist(u, v, w)
    wait([u, v, w])

    logging.info('Mean-removed velocity fields persisted')

    # Assemble the full fluctuation matrix:
    # q = [u'; v'; w']
    q = da.concatenate([u, v, w], axis=0)
    del u, v, w

    dim_q = int(q.shape[0])

    # Segment the time record into nblockf non-overlapping blocks of nDFT
    # snapshots and transform each block: q[p, t] -> q_hat[p, b, f]
    q_blk = q.reshape((dim_q, nblockf, nDFT))
    del q

    q_hat = da.fft.fft(q_blk, axis=2).astype(np.complex64)
    del q_blk

    logging.info('Block FFT computed')

    # Reduced SPOD matrix per frequency:
    # M[f, b, c] = sum_p conj(q_hat[p, b, f]) q_hat[p, c, f]
    Q_hat_f = q_hat.transpose((2, 0, 1))
    del q_hat

    M = da.matmul(da.conj(Q_hat_f).transpose((0, 2, 1)), Q_hat_f)
    del Q_hat_f

    logging.info('Computing reduced SPOD matrices')

    # M is small after the spatial contraction: (nDFT, nblockf, nblockf)
    M = np.asarray(M.compute(), dtype=np.complex128)

    logging.info('Solving SPOD eigenproblems')

    spod_s, spod_v = np.linalg.eigh(M)

    # np.linalg.eigh returns ascending eigenvalues; store them descending
    spod_s = spod_s[:, ::-1]
    spod_v = spod_v[:, :, ::-1]

    freq = np.fft.fftfreq(nDFT, d=1.0)

    logging.info(f"spod_s shape = {spod_s.shape}")
    logging.info(f"spod_v shape = {spod_v.shape}")

    # Save eigenvalues and eigenvectors; spod_v is the temporal basis used
    # by the SPOD triadic step
    np.savez(
        os.path.join(args.output_dir, f'spod_s_{snaps}.npz'),
        spod_s=spod_s, freq=freq, nblockf=nblockf, nDFT=nDFT, snaps=snaps
    )

    np.savez(
        os.path.join(args.output_dir, f'spod_v_{snaps}.npz'),
        spod_v=spod_v, freq=freq, nblockf=nblockf, nDFT=nDFT, snaps=snaps
    )

    logging.info('SPOD Finished')
    logging.info('Completed')
