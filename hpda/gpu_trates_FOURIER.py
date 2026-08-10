import gc
import glob
import logging
import os
import re
from collections import defaultdict

import cupy as cp
import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import get_client, wait

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_node_workers(client):
    """
    Group Dask workers by node address.

    This is used to distribute GPU tasks across the available nodes and devices.
    """
    worker_addresses = list(client.scheduler_info()['workers'].keys())
    node_workers = defaultdict(list)

    for worker in worker_addresses:
        ip_match = re.match(r"tcp://([\d\.]+):\d+", worker)
        if ip_match:
            ip = ip_match.group(1)
            node_workers[ip].append(worker)

    worker_node_map = {w: node for node, workers in node_workers.items() for w in workers}
    return sorted(set(worker_node_map.values()))


def _compute_trates_component_fourier_gpu(component, phi_u, phi_v, phi_w, vol, d, dump_dir):
    """
    Compute the spatial kernel contribution of one transported component
    on the unitary Fourier basis.

    On a Fourier basis the temporal triple product restricts the donor index
    to n = (l + m) mod nt, so only nt^2 of the nt^3 spatial contractions
    survive. The kernel evaluates only those: for each receiver l and
    mediator m it contracts against the conjugate donor gradient at
    n = (l + m) mod nt, giving an (nt, nt) map per spatial block instead of
    an (nt, nt, nt) cube. The conjugate is taken on the differentiated
    (donor) mode, consistent with the SPOD variant.

    The modular donor index is obtained without rolling: the conjugate
    gradient arrays are doubled along the mode axis, and the slice
    [..., m:m + nt] read at position l is entry (l + m) mod nt.

    Each spatial block is processed on one GPU and written temporarily to
    disk. The block files are combined later in the main script.
    """
    logger.info(f'START TRANSFER RATES (GPU BLOCKWISE, {component.upper()}-COMPONENT, FOURIER)')

    client = get_client()
    active_nodes = _get_node_workers(client)

    n_gpu = cp.cuda.runtime.getDeviceCount()
    gpu_slots = [(node, dev) for node in active_nodes for dev in range(n_gpu)]

    total_gpus = len(gpu_slots)
    total_blocks = np.prod(phi_u.numblocks[:3])
    block_indices = list(np.ndindex(*phi_u.numblocks[:3]))

    @delayed
    def resonant_gpu_block(phi_u_block, phi_v_block, phi_w_block, vol_block, d_block,
                           device=0, file_index=None, output_dir='.'):
        with cp.cuda.Device(device):
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            phi_u_gpu = cp.asarray(phi_u_block, dtype=cp.complex64)
            phi_v_gpu = cp.asarray(phi_v_block, dtype=cp.complex64)
            phi_w_gpu = cp.asarray(phi_w_block, dtype=cp.complex64)
            vol_gpu = cp.asarray(vol_block, dtype=cp.float32)
            d_gpu = cp.asarray(d_block, dtype=cp.float32)

            nt = int(phi_u_gpu.shape[3])

            phi_c_gpu = {'u': phi_u_gpu, 'v': phi_v_gpu, 'w': phi_w_gpu}[component]
            phi_c_vol_gpu = phi_c_gpu * vol_gpu

            dtdi, dtdj, dtdk = cp.gradient(phi_c_gpu, axis=(0, 1, 2))
            dtdx_gpu = dtdi * d_gpu[..., 0][..., None] + dtdj * d_gpu[..., 3][..., None] + dtdk * d_gpu[..., 6][..., None]
            dtdy_gpu = dtdi * d_gpu[..., 1][..., None] + dtdj * d_gpu[..., 4][..., None] + dtdk * d_gpu[..., 7][..., None]
            dtdz_gpu = dtdi * d_gpu[..., 2][..., None] + dtdj * d_gpu[..., 5][..., None] + dtdk * d_gpu[..., 8][..., None]

            del dtdi, dtdj, dtdk
            cp.get_default_memory_pool().free_all_blocks()

            # Conjugate donor gradients, doubled along the mode axis so that
            # the slice [..., m:m + nt] indexed by l is entry (l + m) mod nt
            gx2 = cp.concatenate([cp.conj(dtdx_gpu)] * 2, axis=3)
            del dtdx_gpu
            gy2 = cp.concatenate([cp.conj(dtdy_gpu)] * 2, axis=3)
            del dtdy_gpu
            gz2 = cp.concatenate([cp.conj(dtdz_gpu)] * 2, axis=3)
            del dtdz_gpu
            cp.get_default_memory_pool().free_all_blocks()

            result_gpu = cp.zeros((nt, nt), dtype=cp.complex64)

            for m in range(nt):
                donor = (
                    phi_u_gpu[:, :, :, m, None] * gx2[:, :, :, m:m + nt]
                    + phi_v_gpu[:, :, :, m, None] * gy2[:, :, :, m:m + nt]
                    + phi_w_gpu[:, :, :, m, None] * gz2[:, :, :, m:m + nt]
                )
                result_gpu[:, m] = cp.einsum('ijkl,ijkl->l', phi_c_vol_gpu, donor)
                del donor

            result_cpu = cp.asnumpy(result_gpu).astype(np.complex64)

            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f"{component}_output_block_{file_index:04d}.npy"), result_cpu)

            del result_cpu, result_gpu
            del phi_u_gpu, phi_v_gpu, phi_w_gpu, vol_gpu, d_gpu
            del phi_c_gpu, phi_c_vol_gpu, gx2, gy2, gz2

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            return None

    batch = []
    batch_workers = []

    for i, idx in enumerate(block_indices):
        blocks = tuple(idx + (0,))
        worker, device = gpu_slots[i % total_gpus]

        task = resonant_gpu_block(
            phi_u.blocks[blocks],
            phi_v.blocks[blocks],
            phi_w.blocks[blocks],
            vol.blocks[blocks],
            d.blocks[blocks],
            device=device,
            file_index=i,
            output_dir=dump_dir
        )

        batch.append(task)
        batch_workers.append(worker)

        if (i + 1) % total_gpus == 0 or (i + 1) == total_blocks:
            futures = [
                client.compute(task, workers=[worker], allow_other_workers=False)
                for task, worker in zip(batch, batch_workers)
            ]
            wait(futures)
            batch = []
            batch_workers = []

    logger.info('All GPU blocks computed and saved to disk.')
    return None


def compute_trates_u_fourier_gpu(phi_u, phi_v, phi_w, vol, d, dump_dir):
    """Compute the contribution associated with the u equation."""
    return _compute_trates_component_fourier_gpu('u', phi_u, phi_v, phi_w, vol, d, dump_dir)


def compute_trates_v_fourier_gpu(phi_u, phi_v, phi_w, vol, d, dump_dir):
    """Compute the contribution associated with the v equation."""
    return _compute_trates_component_fourier_gpu('v', phi_u, phi_v, phi_w, vol, d, dump_dir)


def compute_trates_w_fourier_gpu(phi_u, phi_v, phi_w, vol, d, dump_dir):
    """Compute the contribution associated with the w equation."""
    return _compute_trates_component_fourier_gpu('w', phi_u, phi_v, phi_w, vol, d, dump_dir)


@delayed
def _read_npy_file_fourier(filename):
    """Read one temporary NumPy block written during the GPU triadic step."""
    return np.load(filename)


def load_and_sum_blocks_lazy_fourier(dump_dir, component):
    """Lazily read all temporary complex block maps for one component and sum them."""
    file_pattern = os.path.join(dump_dir, f"{component}_output_block_*.npy")
    file_list = sorted(glob.glob(file_pattern))

    sample_shape = np.load(file_list[0], mmap_mode="r").shape

    lazy_arrays = [
        da.from_delayed(_read_npy_file_fourier(f), shape=sample_shape, dtype=np.complex64)
        for f in file_list
    ]

    stacked = da.stack(lazy_arrays, axis=0)
    stacked = stacked.rechunk((1,) + sample_shape)
    summed = stacked.sum(axis=0)

    return summed
