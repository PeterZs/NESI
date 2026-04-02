import json
import sys
import os

import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from options import parse_options
from models import *
from utils.base_alg import sample_by_ray_casting, point_mesh_distances, compute_distances, export_points_with_values, eval_views, ray_cast_by_embree
import trimesh


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)

    app_group = parser.add_argument_group('app')
    app_group.add_argument('--gt-samples', type=str, default='./',
                           help='Directory to the evaluated meshes')
    app_group.add_argument('--gt-mesh', type=str, default='./test_data/bimba.obj',
                           help='Directory to the evaluated meshes')
    app_group.add_argument('--n-samples', type=int, default=5_000_000, help='The number of samples per view.')

    args = parser.parse_args()

    model_path = args.model_path

    args.__dict__.update(json.load(open(os.path.join(args.model_path, "args.json"))))

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    state_dict = torch.load(os.path.join(model_path, "model.pth"))

    nesi = NESI(**vars(args))
    if args.jit:
        nesi = torch.jit.script(nesi)

    nesi.load_state_dict(state_dict)

    nesi.to(device)
    nesi.eval()

    total_parameters = sum(p.numel() for p in nesi.parameters())

    print("Total number of parameters: {}".format(total_parameters))

    # Make output directory
    out_dir = os.path.join(model_path, "eval")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    meshes = eval_views(nesi, device, 800)
    for view_i in range(len(meshes)):
        meshes[view_i].export(os.path.join(out_dir, "mesh_" + str(view_i) + ".ply"))

    # Load GT mesh
    gt_mesh = trimesh.load_mesh(args.gt_mesh, force='mesh')
    # Load GT samples
    try:
        gt_samples = trimesh.load_mesh(args.gt_samples).vertices
    except:
        print("No GT samples provided!!!")
        gt_samples = ray_cast_by_embree(gt_mesh, args.n_samples, verbose=True)[:args.n_samples]

    samples = sample_by_ray_casting(nesi.occupancy_check, args.n_samples, 4_000, verbose=True).cpu().numpy()

    # # Output the sample points
    print("Computing distances between sample points and the GT mesh...")
    distances = point_mesh_distances(samples, gt_mesh)
    print("Done.")
    export_points_with_values(samples, distances, os.path.join(out_dir, "nesi_samples.ply"))

    print("Computing numerical results...")
    metrics = compute_distances(samples, gt_samples)
    print("Done.")
    metrics['exp_name'] = os.path.basename(args.gt_samples).split('.')[0]
    metrics['n_views'] = len(args.r_mat)
    metrics['n_params'] = total_parameters

    print("Chamfer distance: {}".format(metrics['Chamfer']))

    with open(os.path.join(out_dir, "numerical_res.json"), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)


