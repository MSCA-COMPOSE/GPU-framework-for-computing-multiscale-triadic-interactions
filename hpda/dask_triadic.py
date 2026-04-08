import argparse
import logging
import os
import socket
import time

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, wait

from hpda_utils import *
from gpu_trates import *

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
    client.upload_file('../hpda/gpu_trates.py')

    print(dask.config.config)

    startTime = time.time()

    # Print a forwarding hint for the Dask dashboard
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "login.cluster.address"
    current_user = os.environ.get('USER')
    logging.info(f"ssh -N -L {port}:{host}:{port} {current_user}@{login_node_address}")
    logging.info('STARTING TRANSFER RATES')

    sequence_file = os.path.join(args.input_dir, args.sequence)
    seq = np.genfromtxt(sequence_file, dtype=int)
    nt = args.nb_r

    logging.info(f"Number of snapshots: {nt}")

    SUBSPACE_NAME = "FLOW_phys"
    blockid = 2
    dump_dir = os.path.join(args.output_dir, 'Dump_0')

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
    vol = da.from_array(vol, chunks=(nx_chunk, ny_chunk, nz_chunk, -1))

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

    ded = np.concatenate([dix, diy, diz, djx, djy, djz, dkx, dky, dkz], axis=1)

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
        da_temp = da.from_delayed(item, dtype=np.float64, shape=(nx, ny, nz, 5))
        Us.append(da_temp[:, :, :, 1][..., np.newaxis])
        Vs.append(da_temp[:, :, :, 2][..., np.newaxis])
        Ws.append(da_temp[:, :, :, 3][..., np.newaxis])

    u = da.concatenate(Us, axis=3).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    v = da.concatenate(Vs, axis=3).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    w = da.concatenate(Ws, axis=3).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))

    for item in lazy_read:
        del item

    # Reload the mean fields produced by the POD step and remove them
    u_mean = dd.read_parquet(
        os.path.join(args.output_dir, f'mean_u_{blockid}.parquet'),
        engine='pyarrow'
    ).to_dask_array(lengths=True)

    v_mean = dd.read_parquet(
        os.path.join(args.output_dir, f'mean_v_{blockid}.parquet'),
        engine='pyarrow'
    ).to_dask_array(lengths=True)

    w_mean = dd.read_parquet(
        os.path.join(args.output_dir, f'mean_w_{blockid}.parquet'),
        engine='pyarrow'
    ).to_dask_array(lengths=True)

    u = u - u_mean.reshape((nx, ny, nz, 1))
    v = v - v_mean.reshape((nx, ny, nz, 1))
    w = w - w_mean.reshape((nx, ny, nz, 1))

    log_variable_details(u)
    logging.info('Persisting matrices')

    # Load the right singular vectors from the POD step
    svd_v_path = os.path.join(args.output_dir, f'svd_v_{nt}.parquet')
    svd_v_df = dd.read_parquet(svd_v_path, engine='pyarrow')
    svd_v = svd_v_df.to_dask_array(lengths=True)

    svd_vt = da.transpose(svd_v)
    svd_vt = svd_vt[:nt, :nt]
    svd_vt, = dask.persist(svd_vt)
    wait([svd_vt])

    u, v, w = dask.persist(u, v, w)
    wait([u, v, w])

    logging.info('CONSTRUCT MODES')

    # Build the spatial POD modes from the fluctuation fields and svd_v
    phi_u = da.einsum('ilmj,jk->ilmk', u, svd_vt)
    del u

    phi_v = da.einsum('ilmj,jk->ilmk', v, svd_vt)
    del v

    phi_w = da.einsum('ilmj,jk->ilmk', w, svd_vt)
    del w

    phi_u = phi_u.rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    phi_v = phi_v.rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    phi_w = phi_w.rechunk((nx_chunk, ny_chunk, nz_chunk, -1))

    phi_u, phi_v, phi_w = dask.persist(phi_u, phi_v, phi_w)
    wait([phi_u, phi_v, phi_w])

    logging.info('COMPUTE TRANSFER RATES')

    # Temporal triple product, averaged over the number of snapshots
    triple_svd = da.einsum('lt,mt,nt->lmn', svd_v, svd_v, svd_v) / nt
    triple_svd = triple_svd.rechunk((nt, nt, nt))
    triple_svd = dask.persist(triple_svd)[0]
    wait([triple_svd])

    logging.info('Triple SVD product computed and persisted')

    # Compute and assemble the contribution from the u equation
    compute_trates_u_gpu(phi_u, phi_v, phi_w, svd_vt, vol, ddd, triple_svd, dump_dir)
    trate_tot_u = load_and_sum_blocks_lazy(dump_dir, 'u', triple_svd=triple_svd)
    trate_tot_u = dask.persist(trate_tot_u)[0]
    wait([trate_tot_u])
    cleanup_component_dump(dump_dir, 'u')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Compute and assemble the contribution from the v equation
    compute_trates_v_gpu(phi_u, phi_v, phi_w, svd_vt, vol, ddd, triple_svd, dump_dir)
    trate_tot_v = load_and_sum_blocks_lazy(dump_dir, 'v', triple_svd=triple_svd)
    trate_tot_v = dask.persist(trate_tot_v)[0]
    wait([trate_tot_v])
    cleanup_component_dump(dump_dir, 'v')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Compute and assemble the contribution from the w equation
    compute_trates_w_gpu(phi_u, phi_v, phi_w, svd_vt, vol, ddd, triple_svd, dump_dir)
    trate_tot_w = load_and_sum_blocks_lazy(dump_dir, 'w', triple_svd=triple_svd)
    trate_tot_w = dask.persist(trate_tot_w)[0]
    wait([trate_tot_w])
    cleanup_component_dump(dump_dir, 'w')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Sum the three contributions and save the final tensor
    trate_tot = trate_tot_u + trate_tot_v + trate_tot_w
    trate_tot = dask.persist(trate_tot)[0]
    wait([trate_tot])

    with open(os.path.join(args.output_dir, 'py_tr_tot_0_2.npy'), 'wb') as sfile_save:
        np.save(sfile_save, trate_tot.compute())

    logging.info('Completed')