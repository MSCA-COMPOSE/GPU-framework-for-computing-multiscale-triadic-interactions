import argparse
import logging
import os
import socket
import time

import dask
import dask.array as da
import numpy as np
from dask.distributed import Client, wait

from hpda_utils import *
from gpu_trates_FOURIER import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

if __name__ == '__main__':

    # Spanwise domain length used in the volume calculation
    L_Z = 0.3

    # Chunking used for the reduced test case during the triadic step
    nx_chunk, ny_chunk, nz_chunk = [50, 50, 128]

    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", help="DASK scheduler file")
    parser.add_argument("input_dir", help="Directory containing input raw files")
    parser.add_argument("sequence", help="Sequence file listing the snapshots to read")
    parser.add_argument("output_dir", help="Directory where the triadic outputs will be written")
    parser.add_argument("nb_r", type=int, help="Number of snapshots to process")
    args = parser.parse_args()

    logging.info('Starting main')

    # Connect to the Dask scheduler created by the submission script
    client = Client(scheduler_file=args.scheduler)

    # Send the helper modules to the workers
    client.upload_file('../hpda/hpda_utils.py')
    client.upload_file('../hpda/gpu_trates_FOURIER.py')

    print(dask.config.config)

    startTime = time.time()

    # Print a forwarding hint for the Dask dashboard
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "login.cluster.address"
    current_user = os.environ.get('USER')
    logging.info(f"ssh -N -L {port}:{host}:{port} {current_user}@{login_node_address}")
    logging.info('STARTING FOURIER TRANSFER RATES')

    sequence_file = os.path.join(args.input_dir, args.sequence)
    seq = np.genfromtxt(sequence_file, dtype=int)
    nt = args.nb_r

    logging.info(f"Number of snapshots: {nt}")

    SUBSPACE_NAME = "FLOW_phys"
    blockid = 2
    dump_dir = os.path.join(args.output_dir, 'Dump_FOURIER_0')

    # The Fourier basis is the unitary discrete Fourier transform of the
    # full snapshot record, phi = fft(q, axis=time) / sqrt(nt), with one
    # coefficient per frequency. No precomputed basis file is required, so
    # this variant has no separate basis-computation stage.
    #
    # Temporal triple product on the Fourier basis, evaluated in closed
    # form: it is non-zero only for frequency triples satisfying
    # n = (l + m) mod nt, the circular-convolution form of the zero-sum
    # condition, where it takes the constant value 1 / sqrt(nt). This is
    # the SPOD triple product of dask_triadic_SPOD.py in the single-block
    # limit (nblockf = 1, theta = 1), with the conjugate convention
    # matching the spatial kernel phi_l phi_m conj(grad phi_n). Because the
    # non-resonant entries vanish identically, the spatial kernel is
    # evaluated only on the resonant set and the triple product reduces to
    # the scalar factor applied below.
    triple_fourier = 1.0 / np.sqrt(np.float32(nt))

    logging.info('Reading mesh')

    # Read the structured grid for the selected block
    gridname = f"{SUBSPACE_NAME}_GRID_{blockid}.xyz"
    nx, ny, nz = read_grid_header(os.path.join(args.input_dir, gridname))
    totdim = nx * ny * nz

    xyz = read_grid(os.path.join(args.input_dir, gridname), nx, ny, nz)
    logging.info(xyz.shape)

    # Compute the cell volumes from the mesh
    vol_all = compute_grid_volume(xyz, L_Z)
    logging.info(vol_all.shape)

    vol = vol_all[:, :, :, np.newaxis]
    vol = da.from_array(vol.astype(np.float32), chunks=(nx_chunk, ny_chunk, nz_chunk, -1))

    # Compute mesh derivatives and build the inverse metric terms
    dxi, dxj, dxk = np.gradient(xyz[:, :, :, 0], edge_order=2)
    dyi, dyj, dyk = np.gradient(xyz[:, :, :, 1], edge_order=2)
    dzi, dzj, dzk = np.gradient(xyz[:, :, :, 2], edge_order=2)

    logging.info('Mesh gradient')

    dxi = dxi.reshape(totdim, 1)
    dxj = dxj.reshape(totdim, 1)
    dxk = dxk.reshape(totdim, 1)
    dyi = dyi.reshape(totdim, 1)
    dyj = dyj.reshape(totdim, 1)
    dyk = dyk.reshape(totdim, 1)
    dzi = dzi.reshape(totdim, 1)
    dzj = dzj.reshape(totdim, 1)
    dzk = dzk.reshape(totdim, 1)

    logging.info('Reshape 1')

    DetJ = (
        dxi * dyj * dzk
        + dxj * dyk * dzi
        + dxk * dyi * dzj
        - dxi * dyk * dzj
        - dxj * dyi * dzk
        - dxk * dyj * dzi
    )

    dix = 1 / DetJ * (dyj * dzk - dyk * dzj)
    diy = -1 / DetJ * (dxj * dzk - dxk * dzj)
    diz = 1 / DetJ * (dxj * dyk - dyj * dxk)

    djx = -1 / DetJ * (dyi * dzk - dzi * dyk)
    djy = 1 / DetJ * (dxi * dzk - dzi * dxk)
    djz = -1 / DetJ * (dxi * dyk - dxk * dyi)

    dkx = 1 / DetJ * (dyi * dzj - dzi * dyj)
    dky = -1 / DetJ * (dxi * dzj - dxj * dzi)
    dkz = 1 / DetJ * (dxi * dyj - dxj * dyi)

    ded = np.concatenate([dix, diy, diz, djx, djy, djz, dkx, dky, dkz], axis=1).astype(np.float32)

    ddd = da.from_array(
        ded.reshape(nx, ny, nz, 9),
        chunks=(nx_chunk, ny_chunk, nz_chunk, -1)
    )

    ddd, = dask.persist(ddd)
    wait([ddd])

    del ded
    logging.info('Det Finished')

    # Build the list of raw snapshot files to read
    files = []
    for num in seq[:nt]:
        filename = f"{SUBSPACE_NAME}_{blockid}_{num}.raw"
        files.append(os.path.join(args.input_dir, filename))

    lazy_read = [read_file(filename, nx, ny, nz) for filename in files]
    logging.info('Started Read')

    # Assemble the three velocity components as
    # (nx, ny, nz, time) arrays
    Us = []
    Vs = []
    Ws = []

    for item in lazy_read:
        da_temp = da.from_delayed(item, dtype=np.float32, shape=(nx, ny, nz, 5))
        Us.append(da_temp[:, :, :, 1][..., np.newaxis])
        Vs.append(da_temp[:, :, :, 2][..., np.newaxis])
        Ws.append(da_temp[:, :, :, 3][..., np.newaxis])

    u = da.concatenate(Us, axis=3).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    v = da.concatenate(Vs, axis=3).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    w = da.concatenate(Ws, axis=3).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))

    for item in lazy_read:
        del item

    # Remove the temporal mean at each spatial point
    u = (u - da.mean(u, axis=3, keepdims=True)).astype(np.float32)
    v = (v - da.mean(v, axis=3, keepdims=True)).astype(np.float32)
    w = (w - da.mean(w, axis=3, keepdims=True)).astype(np.float32)

    log_variable_details(u)
    logging.info('Persisting matrices')

    u, v, w = dask.persist(u, v, w)
    wait([u, v, w])

    logging.info('CONSTRUCT FOURIER MODES')

    # Unitary two-sided Fourier modes: one coefficient per frequency, with
    # the flattened mode index equal to the frequency index
    phi_u = (da.fft.fft(u, axis=3) / np.sqrt(nt)).astype(np.complex64)
    del u
    phi_v = (da.fft.fft(v, axis=3) / np.sqrt(nt)).astype(np.complex64)
    del v
    phi_w = (da.fft.fft(w, axis=3) / np.sqrt(nt)).astype(np.complex64)
    del w

    phi_u = phi_u.rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    phi_v = phi_v.rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    phi_w = phi_w.rechunk((nx_chunk, ny_chunk, nz_chunk, -1))

    phi_u, phi_v, phi_w = dask.persist(phi_u, phi_v, phi_w)
    wait([phi_u, phi_v, phi_w])

    logging.info('COMPUTE TRANSFER RATES')

    # Compute and assemble the contribution from the u equation
    compute_trates_u_fourier_gpu(phi_u, phi_v, phi_w, vol, ddd, dump_dir)
    map_tot_u = load_and_sum_blocks_lazy_fourier(dump_dir, 'u')
    map_tot_u = dask.persist(map_tot_u)[0]
    wait([map_tot_u])
    cleanup_component_dump(dump_dir, 'u')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Compute and assemble the contribution from the v equation
    compute_trates_v_fourier_gpu(phi_u, phi_v, phi_w, vol, ddd, dump_dir)
    map_tot_v = load_and_sum_blocks_lazy_fourier(dump_dir, 'v')
    map_tot_v = dask.persist(map_tot_v)[0]
    wait([map_tot_v])
    cleanup_component_dump(dump_dir, 'v')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Compute and assemble the contribution from the w equation
    compute_trates_w_fourier_gpu(phi_u, phi_v, phi_w, vol, ddd, dump_dir)
    map_tot_w = load_and_sum_blocks_lazy_fourier(dump_dir, 'w')
    map_tot_w = dask.persist(map_tot_w)[0]
    wait([map_tot_w])
    cleanup_component_dump(dump_dir, 'w')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Sum the three contributions, apply the temporal triple product, and
    # scatter the resonant map onto the (nt, nt, nt) tensor so that the
    # saved object has the same shape and index convention as the POD and
    # SPOD outputs: entry (l, m, n) with n = (l + m) mod nt, zero elsewhere
    map_tot = (map_tot_u + map_tot_v + map_tot_w).compute().astype(np.complex64)
    map_tot *= np.complex64(triple_fourier)

    trate_tot = np.zeros((nt, nt, nt), dtype=np.complex64)
    idx_l = np.arange(nt)[:, None]
    idx_m = np.arange(nt)[None, :]
    trate_tot[idx_l, idx_m, (idx_l + idx_m) % nt] = map_tot

    with open(os.path.join(args.output_dir, 'py_tr_FOURIER_tot_0_2.npy'), 'wb') as sfile_save:
        np.save(sfile_save, trate_tot)

    logging.info('Completed')
