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
from gpu_trates_SPOD import *

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
    parser.add_argument("output_dir", help="Directory containing the SPOD basis and where the triadic outputs will be written")
    parser.add_argument("nb_r", type=int, help="Number of snapshots to process")
    args = parser.parse_args()

    logging.info('Starting main')

    # Connect to the Dask scheduler created by the submission script
    client = Client(scheduler_file=args.scheduler)

    # Send the helper modules to the workers
    client.upload_file('../hpda/hpda_utils.py')
    client.upload_file('../hpda/gpu_trates_SPOD.py')

    print(dask.config.config)

    startTime = time.time()

    # Print a forwarding hint for the Dask dashboard
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "login.cluster.address"
    current_user = os.environ.get('USER')
    logging.info(f"ssh -N -L {port}:{host}:{port} {current_user}@{login_node_address}")
    logging.info('STARTING SPOD TRANSFER RATES')

    sequence_file = os.path.join(args.input_dir, args.sequence)
    seq = np.genfromtxt(sequence_file, dtype=int)
    nt = args.nb_r

    logging.info(f"Number of snapshots: {nt}")

    SUBSPACE_NAME = "FLOW_phys"
    blockid = 2
    dump_dir = os.path.join(args.output_dir, 'Dump_SPOD_0')

    # Load the SPOD temporal basis produced by dask_read_SPOD.py
    spod_v_npz = np.load(os.path.join(args.output_dir, f'spod_v_{nt}.npz'))
    theta_np = spod_v_npz['spod_v'].astype(np.complex64)

    nDFT = int(theta_np.shape[0])
    nblockf = int(theta_np.shape[1])

    logging.info(f"SPOD nDFT = {nDFT}, nblockf = {nblockf}")

    theta = da.from_array(theta_np, chunks=(nDFT, nblockf, nblockf))

    # Temporal triple product on the block-SPOD basis, evaluated in closed
    # form: with the flattening mode = f*nblockf + r, the product is non-zero
    # only for frequency triples satisfying fn = (fl + fm) mod nDFT, the
    # circular-convolution form of the zero-sum condition, and reads
    # (1/nt) sum_b conj(theta[fl,b,rl]) conj(theta[fm,b,rm]) theta[fn,b,rn] / sqrt(nDFT),
    # with the 1/nt factor giving the triple product the scaling of a sampled
    # time average, as for the POD triple product of dask_triadic.py, and the
    # conjugate convention matching the spatial kernel
    # phi_l phi_m conj(grad phi_n).
    triple_np = np.zeros((nt, nt, nt), dtype=np.complex64)

    for fl in range(nDFT):
        for fm in range(nDFT):
            fn = (fl + fm) % nDFT
            block = np.einsum(
                'br,bs,bt->rst',
                np.conj(theta_np[fl, :, :]),
                np.conj(theta_np[fm, :, :]),
                theta_np[fn, :, :],
                optimize=True
            ) / (np.float32(nt) * np.sqrt(np.float32(nDFT)))
            triple_np[
                fl * nblockf:(fl + 1) * nblockf,
                fm * nblockf:(fm + 1) * nblockf,
                fn * nblockf:(fn + 1) * nblockf
            ] = block.astype(np.complex64)

    triple_spod = da.from_array(triple_np, chunks=(nt, nt, nt))
    triple_spod = dask.persist(triple_spod)[0]
    wait([triple_spod])

    logging.info('SPOD triple product computed and persisted')

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

    logging.info('CONSTRUCT SPOD MODES')

    # Segment the time record into blocks, transform each block, and project
    # onto the SPOD temporal basis; the flattened mode index is f*nblockf + r
    u_blk = u.reshape(nx, ny, nz, nblockf, nDFT)
    del u
    v_blk = v.reshape(nx, ny, nz, nblockf, nDFT)
    del v
    w_blk = w.reshape(nx, ny, nz, nblockf, nDFT)
    del w

    u_hat = da.fft.fft(u_blk, axis=4).astype(np.complex64)
    del u_blk
    v_hat = da.fft.fft(v_blk, axis=4).astype(np.complex64)
    del v_blk
    w_hat = da.fft.fft(w_blk, axis=4).astype(np.complex64)
    del w_blk

    phi_u = da.einsum('ijkbf,fbr->ijkfr', u_hat, theta) / np.sqrt(nDFT)
    del u_hat
    phi_v = da.einsum('ijkbf,fbr->ijkfr', v_hat, theta) / np.sqrt(nDFT)
    del v_hat
    phi_w = da.einsum('ijkbf,fbr->ijkfr', w_hat, theta) / np.sqrt(nDFT)
    del w_hat

    phi_u = phi_u.astype(np.complex64).reshape(nx, ny, nz, nt).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    phi_v = phi_v.astype(np.complex64).reshape(nx, ny, nz, nt).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))
    phi_w = phi_w.astype(np.complex64).reshape(nx, ny, nz, nt).rechunk((nx_chunk, ny_chunk, nz_chunk, -1))

    phi_u, phi_v, phi_w = dask.persist(phi_u, phi_v, phi_w)
    wait([phi_u, phi_v, phi_w])

    logging.info('COMPUTE TRANSFER RATES')

    # Compute and assemble the contribution from the u equation
    compute_trates_u_spod_gpu(phi_u, phi_v, phi_w, vol, ddd, dump_dir)
    trate_tot_u = load_and_sum_blocks_lazy_spod(dump_dir, 'u')
    trate_tot_u = dask.persist(trate_tot_u)[0]
    wait([trate_tot_u])
    cleanup_component_dump(dump_dir, 'u')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Compute and assemble the contribution from the v equation
    compute_trates_v_spod_gpu(phi_u, phi_v, phi_w, vol, ddd, dump_dir)
    trate_tot_v = load_and_sum_blocks_lazy_spod(dump_dir, 'v')
    trate_tot_v = dask.persist(trate_tot_v)[0]
    wait([trate_tot_v])
    cleanup_component_dump(dump_dir, 'v')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Compute and assemble the contribution from the w equation
    compute_trates_w_spod_gpu(phi_u, phi_v, phi_w, vol, ddd, dump_dir)
    trate_tot_w = load_and_sum_blocks_lazy_spod(dump_dir, 'w')
    trate_tot_w = dask.persist(trate_tot_w)[0]
    wait([trate_tot_w])
    cleanup_component_dump(dump_dir, 'w')

    logging.info('TIME ELAPSED: {}'.format(time.time() - startTime))

    # Sum the three contributions, apply the temporal triple product,
    # and save the final complex tensor
    trate_tot = (trate_tot_u + trate_tot_v + trate_tot_w) * triple_spod
    trate_tot = trate_tot.astype(np.complex64)
    trate_tot = dask.persist(trate_tot)[0]
    wait([trate_tot])

    with open(os.path.join(args.output_dir, 'py_tr_SPOD_tot_0_2.npy'), 'wb') as sfile_save:
        np.save(sfile_save, trate_tot.compute())

    logging.info('Completed')
