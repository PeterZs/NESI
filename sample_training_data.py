import json
import sys
import os
import numpy as np
import torch
import argparse
import trimesh
from utils.base_alg import training_data_sampling
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation view selection for ESI.')
    parser.add_argument('-i', '--input-views', type=str, default="./test_data/bimba/4/view_directions.txt", help='Input view directions')
    parser.add_argument('-c', '--cutting-plane', type=str, default="./test_data/plane.txt", help='Input cutting plane, if any')
    parser.add_argument('-o', '--output-dir', type=str, default="./test_data/bimba/", help='Save output data to the directory')
    parser.add_argument('--gt-mesh', type=str, default="./test_data/bimba.obj",
                        help='The ground-truth mesh')
    parser.add_argument('--n-jobs', type=int, default=4, help='Number of computing threads.')
    parser.add_argument('--n-samples', type=int, default=5_000_000, help='The number of samples per view.')
    parser.add_argument('--mask', action='store_true',
                        help='The switch to HF mask mode.')
    parser.add_argument('--dbg', action='store_true', help='The switch to debug mode (leading to additional outputs).')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse
    args = get_args()

    # Pick device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read view directions
    view_dirs = np.loadtxt(args.input_views)

    if view_dirs.size == 0:
        exit()

    if len(view_dirs.shape) == 1:
        view_dirs = view_dirs.reshape(1, -1)
    z_origins = [-2] * view_dirs.shape[0]

    if os.path.isfile(args.cutting_plane):
        cutting_plane_spec = np.loadtxt(args.cutting_plane)
        view_dirs = np.vstack((view_dirs, cutting_plane_spec[:3][None, ...], -cutting_plane_spec[:3][None, ...]))
        z_origins += [cutting_plane_spec[3], -cutting_plane_spec[3]]
    # Make output directory
    out_dir = os.path.join(args.output_dir, "training_samples")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load GT mesh
    gt_mesh = trimesh.load_mesh(args.gt_mesh, force='mesh')
    for i, view_dir in tqdm(enumerate(view_dirs)):
        results = training_data_sampling(view_dir, mesh=gt_mesh, z_origin=z_origins[i], n_samples=args.n_samples,
                                         is_dhf=(i == 0), prev_view_dirs=view_dirs[:i], with_mask=args.mask & (z_origins[i] < -1.9))
        training_data = dict()
        training_data['r_mat'] = results[0]
        training_data['surface_samples'] = results[1]
        training_data['surface_normals'] = results[2]
        training_data['spatial_samples'] = results[3]
        training_data['roi_mask'] = results[4]
        training_data['expanded_roi_mask'] = results[5]
        training_data['z_origin'] = z_origins[i]

        np.savez(os.path.join(out_dir, str(i) + '.npz'), **training_data)

        if args.dbg:
            spatial_samples = training_data['spatial_samples'][..., :3]
            spatial_samples[..., 2] = -2.0
            trimesh.Trimesh(spatial_samples @ training_data['r_mat'].T).export(
                os.path.join(out_dir, str(i) + '_spatial_samples.ply'))

            if i == 0:
                trimesh.Trimesh(training_data['surface_samples'][..., :3] @ training_data['r_mat'].T,
                                vertex_colors=training_data['surface_normals'][0] @ training_data[
                                    'r_mat'].T / 2. + 0.5).export(
                    os.path.join(out_dir, str(i) + '_surface_samples_a.ply'))

                trimesh.Trimesh(training_data['surface_samples'][..., [0, 1, 3]] @ training_data['r_mat'].T,
                                vertex_colors=training_data['surface_normals'][1] @ training_data[
                                    'r_mat'].T / 2. + 0.5).export(
                    os.path.join(out_dir, str(i) + '_surface_samples_b.ply'))
            else:
                trimesh.Trimesh(training_data['surface_samples'] @ training_data['r_mat'].T,
                                vertex_colors=training_data['surface_normals'] @ training_data[
                                    'r_mat'].T / 2. + 0.5).export(
                    os.path.join(out_dir, str(i) + '_surface_samples.ply'))

                if training_data['roi_mask'] is not None:
                    v_color = np.zeros_like(training_data['surface_samples'])
                    v_color[:, 0] = 1# R
                    v_color[training_data['expanded_roi_mask'], :] = 0
                    v_color[training_data['expanded_roi_mask'], 2] = 1 # B
                    v_color[training_data['roi_mask'], :] = 0
                    v_color[training_data['roi_mask'], 1] = 1 # G
                    trimesh.Trimesh(training_data['surface_samples'] @ training_data['r_mat'].T,
                                    vertex_colors=v_color).export(os.path.join(out_dir, str(i) + '_roi_vis.ply'))



