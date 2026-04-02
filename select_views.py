import numpy as np
import torch
import argparse
import os
import trimesh
import point_cloud_utils as pcu
from models.esi import ESI
from pykdtree.kdtree import KDTree
import igl
import joblib
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import itertools
import multiprocessing as mp
from utils.base_alg import generate_view_directions, is_visible, fast_hf_sampling
import time
import json


def get_args():
    parser = argparse.ArgumentParser(description='Search optimal view selection for ESI.')
    parser.add_argument('-i', '--input-file', type=str, default="./test_data/bimba.obj", help='Input file name')
    parser.add_argument('-o', '--output-dir', type=str, default="./test_data/bimba", help='Save output data to the directory')
    parser.add_argument('--n-dhf', type=int, default=50, help='Number of DHF views.')
    parser.add_argument('--n-hf', type=int, default=80, help='Number of HF views.')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of computing threads.')
    parser.add_argument('--n-samples', type=int, default=10000, help='The number of samples per view.')
    parser.add_argument('--no-skip', action='store_true', help='Run full experiments on all view numbers.')
    parser.add_argument('--remove-axes', action='store_true', help='No axis directions included.')
    parser.add_argument('--dbg', action='store_true', help='The switch to debug mode (leading to additional outputs).')
    parser.add_argument('--n-views', type=int, default=4, help='Number of the total maximum view directions.')
    parser.add_argument('--angle', action='store_true', help='The switch to the CD+angle mode.')
    parser.add_argument('--using-axes', action='store_true', help='Set the view directions to axes directly')
    parser.add_argument('--angle-threshold', type=float, default=85, help='The threshold of angle score (in degree).')
    parser.add_argument('-p', '--plane-spec', dest='plane_spec', type=str, default="./",
                        help='Input file name if there is a cutting plane.')

    args = parser.parse_args()
    return args


def data_preprocessing(view_id, view_dirs, mesh, n_samples, is_dhf, mesh_samples, shifted_mesh_samples, mesh_sample_normals=None, plane_normals=None, plane_z_origins=None):
    # if plane normal is None, need to compose z_origins
    if plane_normals is None:
        z_origins = np.full(1, -2)
    else:
        z_origins = np.array([-2] + plane_z_origins)

    view_dir = view_dirs[view_id]
    if is_dhf:
        if plane_normals is None:
            esi = ESI(view_dir[np.newaxis, ...], mesh, dhf_ids=[0])
        else:
            esi_view_dirs = np.stack(([view_dir] + plane_normals), axis=0)
            esi = ESI(esi_view_dirs, mesh, dhf_ids=[0], z_origins=z_origins)


    else:
        esi = ESI(view_dir[np.newaxis, ...], mesh)
    #
    # # Generate sample points on ESI
    samples = fast_hf_sampling(esi, n_samples, 5_000, verbose=False)

    tree = KDTree(samples)
    mesh_to_esi_dis, _ = tree.query(mesh_samples.astype(np.float32))

    # check visibility
    visible_mask = is_visible(view_dir, shifted_mesh_samples, esi.intersector, is_dhf=is_dhf)

    # check visibility for planes if applicable
    if plane_normals is not None:
        visible_mask |= is_visible(plane_normals[0], shifted_mesh_samples, esi.intersector, is_dhf=False, z_origin=plane_z_origins[0])
        visible_mask |= is_visible(plane_normals[1], shifted_mesh_samples, esi.intersector, is_dhf=False, z_origin=plane_z_origins[1])

    # set the distance of visible points as 0
    mesh_to_esi_dis[visible_mask] = 0.

    # compute the distance from the ESI hull to the mesh
    esi_to_mesh_dis = np.abs(
        igl.signed_distance(samples.astype(np.float64), mesh.vertices, mesh.faces)[0]) #, return_normals=False

    esi_to_mesh_dis[esi_to_mesh_dis <= 5e-4] = 0
    floaters = samples[esi_to_mesh_dis > 5e-4]
    floater_dis = esi_to_mesh_dis[esi_to_mesh_dis > 5e-4]

    if args.dbg:
        from utils.base_alg import create_arrow
        prefix = "DHF" if is_dhf else "HF"
        trimesh.Trimesh(samples).export(os.path.join(args.output_dir, "vis", prefix + "_" + str(view_id) + ".ply"))
        if is_dhf:
            trimesh.util.concatenate(create_arrow(view_dir), create_arrow(-view_dir)).export(
                os.path.join(args.output_dir, "vis", prefix + "_" + str(view_id) + "_arrow.ply"))
        else:
            create_arrow(view_dir).export(
                os.path.join(args.output_dir, "vis", prefix + "_" + str(view_id) + "_arrow.ply"))

    if not args.angle:
        return mesh_to_esi_dis, esi_to_mesh_dis, floaters, floater_dis

    # compute the view angle score
    angle_cos = np.abs((view_dir * mesh_sample_normals).sum(-1))
    angle_cos[angle_cos > 1] = 1
    angles = np.arccos(angle_cos) / np.pi * 180.
    angle_scores = np.tanh(angles - args.angle_threshold) * 0.5 + 0.5

    angle_scores[~visible_mask] = 1

    if args.dbg:
        v_colors = np.zeros_like(mesh_samples)
        v_colors[angle_scores == 0] = np.array([0., 1., 0.]) #Green
        v_colors[angle_scores == 1] = np.array([1., 1., 0.]) #Yellow
        v_colors[~visible_mask] = np.array([1., 0., 0.]) #Red
        trimesh.Trimesh(mesh_samples, vertex_colors=v_colors).export(
            os.path.join(args.output_dir, "vis", prefix + "_" + str(view_id) + "_angle_score.ply"))

    return mesh_to_esi_dis, esi_to_mesh_dis, floaters, floater_dis, angle_scores


def compute_chamfer(view_ids, mesh_to_dhf_dis, mesh_to_hf_dis, dhf_floater_dis, hf_floater_dis, dhf_floaters_to_hf_visibility,
                    hf_floaters_to_dhf_visibility, hf_floaters_to_hf_visibility, n_samples, dhf_angle_scores=None, hf_angle_scores=None):

    dhf_id = view_ids[0]
    hf_ids = view_ids[1:]

    # Compute mesh to ESI distance as the minimum distances from points to all ESI view directions
    mesh_to_esi_dis = np.concatenate([mesh_to_dhf_dis[dhf_id][None, ...], mesh_to_hf_dis[hf_ids]])
    mesh_to_esi_dis = np.min(mesh_to_esi_dis, axis=0)
    mesh_to_esi_dis = mesh_to_esi_dis.mean()

    # Find invisible floater points and add them together
    floater_visible_mask = np.zeros(dhf_floater_dis[dhf_id].shape[0], dtype=bool)
    for k in hf_ids:
        floater_visible_mask = floater_visible_mask | dhf_floaters_to_hf_visibility[(dhf_id, k)]
    floater_dis = dhf_floater_dis[dhf_id][~floater_visible_mask].sum() / n_samples

    for x in hf_ids:
        floater_visible_mask = hf_floaters_to_dhf_visibility[(x, dhf_id)]
        for y in hf_ids:
            if x != y:
                floater_visible_mask = floater_visible_mask | hf_floaters_to_hf_visibility[(x, y)]
        floater_dis += hf_floater_dis[x][~floater_visible_mask].sum() / n_samples

    if args.angle:
        angles_scores = np.concatenate([dhf_angle_scores[dhf_id][None, ...], hf_angle_scores[hf_ids]])
        angles_scores = np.min(angles_scores, axis=0)
        angles_scores = angles_scores.mean()# ** args.angle_power
        return mesh_to_esi_dis + floater_dis + min((mesh_to_esi_dis + floater_dis + 1e-4) * angles_scores, 1e-3)
    else:
        return mesh_to_esi_dis + floater_dis


def find_best_hfs(dhf_id, all_hf_combinations, shared_min_cd, mesh_to_hf_dis, dhf_floater_dis, hf_floater_dis,
                  dhf_floaters_to_hf_visibility,
                  hf_floaters_to_dhf_visibility, hf_floaters_to_hf_visibility, n_samples, angle_scores=None):

    local_min_cd = min(shared_min_cd)

    best_views = []
    pruning_count = 0

    # create a matrix for acceleration: consider only non-zero points in mesh-DHF distance matrix
    mesh_to_esi_samples = float(args.n_samples)

    for hf_view_set in all_hf_combinations:

        # Compute mesh to ESI distance as the minimum distances from points to all ESI view directions
        mesh_to_esi_dis = np.min(mesh_to_hf_dis[list(hf_view_set) + [-1]], axis=0).sum() / mesh_to_esi_samples
        if args.angle:
            esi_angle_score = (np.min(angle_scores[list(hf_view_set) + [-1]], axis=0).sum() / mesh_to_esi_samples)# ** args.angle_power
            if mesh_to_esi_dis + min((mesh_to_esi_dis + 1e-4) * esi_angle_score, 1e-3) >= local_min_cd:
                pruning_count += 1
                continue
        else:
            if mesh_to_esi_dis >= local_min_cd:
                pruning_count += 1
                continue

        # Find invisible floater points and add them together
        floater_visible_mask = np.zeros(dhf_floater_dis[dhf_id].shape[0], dtype=bool)
        for k in hf_view_set:
            floater_visible_mask = floater_visible_mask | dhf_floaters_to_hf_visibility[(dhf_id, k)]
        floater_dis = dhf_floater_dis[dhf_id][~floater_visible_mask].sum() / n_samples

        if args.angle:
            if mesh_to_esi_dis + floater_dis + min((mesh_to_esi_dis + floater_dis + 1e-4) * esi_angle_score, 1e-3) >= local_min_cd:
                continue
        else:
            if (mesh_to_esi_dis + floater_dis) >= local_min_cd:
                continue

        for x in hf_view_set:
            floater_visible_mask = hf_floaters_to_dhf_visibility[(x, dhf_id)]
            for y in hf_view_set:
                if x != y:
                    floater_visible_mask = floater_visible_mask | hf_floaters_to_hf_visibility[(x, y)]
            floater_dis += hf_floater_dis[x][~floater_visible_mask].sum() / n_samples

            if args.angle:
                if mesh_to_esi_dis + floater_dis + min((mesh_to_esi_dis + floater_dis + 1e-4) * esi_angle_score, 1e-3) >= local_min_cd:
                    break
            else:
                if (mesh_to_esi_dis + floater_dis) >= local_min_cd:
                    break

        if args.angle:
            if mesh_to_esi_dis + floater_dis + min((mesh_to_esi_dis + floater_dis + 1e-4) * esi_angle_score, 1e-3) < local_min_cd:
                local_min_cd = mesh_to_esi_dis + floater_dis + min((mesh_to_esi_dis + floater_dis + 1e-4) * esi_angle_score, 1e-3)
                best_views = [dhf_id, *hf_view_set]
        else:
            if (mesh_to_esi_dis + floater_dis) < local_min_cd:
                local_min_cd = mesh_to_esi_dis + floater_dis
                best_views = [dhf_id, *hf_view_set]

    if shared_min_cd[dhf_id] > local_min_cd:
        shared_min_cd[dhf_id] = local_min_cd

    return best_views, local_min_cd, pruning_count


def select_views(mesh):

    log = dict()
    log['exp_name'] = os.path.basename(args.input_file).split('.')[0]
    log['n_vertices'] = mesh.vertices.shape[0]
    log['n_faces'] = mesh.faces.shape[0]
    log['n_dhf'] = args.n_dhf
    log['n_hf'] = args.n_hf
    log['threads'] = args.n_jobs
    log['use_angle'] = args.angle
    log['angle_threshold'] = args.angle_threshold

    if os.path.isfile(args.plane_spec):
        plane_spec = json.load(open(args.plane_spec))['cutting_planes'][0]
        plane_normal = np.array(plane_spec['normal']) / np.linalg.norm(np.array(plane_spec['normal']))
        plane_point_wc = np.array(plane_spec['cutting_point'])

        # plane z origin is D = Ax + By + Cz if normal (A, B, C) are normalized
        plane_z_origins = [np.dot(plane_normal, plane_point_wc), -np.dot(plane_normal, plane_point_wc)]
        plane_normals = [plane_normal, -plane_normal]
    else:
        plane_z_origins = None
        plane_normals = None

    start_wall_time = time.time()
    start_process_time = time.process_time_ns()

    # sampling on the ground truth mesh to compute CD
    mesh_samples, mesh_sample_fids = trimesh.sample.sample_surface(mesh, args.n_samples)

    # potential robust issue for degenerated faces
    mesh_normals = mesh.face_normals[mesh_sample_fids]

    # shift all sample points outside for checking visibility
    shifted_mesh_samples = mesh_samples + mesh_normals * 1e-5

    dhf_view_dirs = generate_view_directions(args.n_dhf, is_dhf=True, include_axes=not args.remove_axes)
    hf_view_dirs = generate_view_directions(args.n_hf, is_dhf=False, include_axes=not args.remove_axes)

    #############################################
    # Precompute data for DHF views
    dhf_preprocessing_func = partial(data_preprocessing, view_dirs=dhf_view_dirs, mesh=mesh, n_samples=args.n_samples,
                                     is_dhf=True, mesh_samples=mesh_samples, shifted_mesh_samples=shifted_mesh_samples,
                                     mesh_sample_normals=mesh_normals, plane_normals=plane_normals, plane_z_origins=plane_z_origins)
    if args.n_jobs > 1:
        print(joblib.cpu_count(), "CPUs available")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(dhf_preprocessing_func)(view_id=i) for i in tqdm(range(args.n_dhf)))
    else:
        print("Single threaded preprocessing")
        results = [dhf_preprocessing_func(view_id=i) for i in tqdm(range(args.n_dhf))]
    mesh_to_dhf_dis = [result[0] for result in results]
    dhf_to_mesh_dis = [result[1] for result in results]
    dhf_floaters = [result[2] for result in results]
    dhf_floater_dis = [result[3] for result in results]
    if args.angle:
        dhf_angle_scores = [result[4] for result in results]
        dhf_angle_scores = np.stack(dhf_angle_scores)

    mesh_to_dhf_dis = np.stack(mesh_to_dhf_dis)
    dhf_to_mesh_dis = np.stack(dhf_to_mesh_dis)

    #############################################
    # Precompute data for HF views
    hf_preprocessing_func = partial(data_preprocessing, view_dirs=hf_view_dirs, mesh=mesh, n_samples=args.n_samples,
                                    is_dhf=False,
                                    mesh_samples=mesh_samples, shifted_mesh_samples=shifted_mesh_samples, mesh_sample_normals=mesh_normals)
    if args.n_jobs > 1:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(hf_preprocessing_func)(view_id=i) for i in tqdm(range(args.n_hf)))
    else:
        results = [hf_preprocessing_func(view_id=i) for i in tqdm(range(args.n_hf))]
    mesh_to_hf_dis = [result[0] for result in results]

    hf_floaters = [result[2] for result in results]
    hf_floater_dis = [result[3] for result in results]
    if args.angle:
        hf_angle_scores = [result[4] for result in results]
        hf_angle_scores = np.stack(hf_angle_scores)

    mesh_to_hf_dis = np.stack(mesh_to_hf_dis)

    # Create an auxiliary list of array for acceleration
    # Consider only non-zero points in mesh-DHF distance matrix
    aux_mesh_dis_lst = []
    for i in range(args.n_dhf):
        tmp_mat = np.concatenate([mesh_to_hf_dis, mesh_to_dhf_dis[i][None, ...]])
        aux_mesh_dis_lst += [tmp_mat[:, mesh_to_dhf_dis[i] > 0]]#

    if args.angle:
        aux_angle_scores_lst = []
        for i in range(args.n_dhf):
            tmp_mat = np.concatenate([hf_angle_scores, dhf_angle_scores[i][None, ...]])
            aux_angle_scores_lst += [tmp_mat[:, dhf_angle_scores[i] > 0]]
    else:
        aux_angle_scores_lst = [None] * args.n_dhf

    #############################################
    # Precompute data for floater visibility
    dhf_floaters_to_hf_visibility = dict()
    hf_floaters_to_hf_visibility = dict()
    for i in tqdm(range(args.n_hf)):
        esi = ESI(hf_view_dirs[i][np.newaxis, ...], mesh)
        for k in range(args.n_dhf):
            is_inside = esi.occupancy_check(dhf_floaters[k])
            dhf_floaters_to_hf_visibility[(k, i)] = ~is_inside
        for k in range(args.n_hf):
            is_inside = esi.occupancy_check(hf_floaters[k])
            hf_floaters_to_hf_visibility[(k, i)] = ~is_inside

    hf_floaters_to_dhf_visibility = dict()
    for i in tqdm(range(args.n_dhf)):
        if plane_normals is None:
            esi = ESI(dhf_view_dirs[i][np.newaxis, ...], mesh, dhf_ids=[0])
        else:
            # view directions for normal view and plane view
            esi_view_dirs = np.stack(([dhf_view_dirs[i]] + plane_normals), axis=0)
            esi = ESI(esi_view_dirs, mesh, dhf_ids=[0], z_origins=np.array([-2] + plane_z_origins))

        for k in range(args.n_hf):
            is_inside = esi.occupancy_check(hf_floaters[k])
            hf_floaters_to_dhf_visibility[(k, i)] = ~is_inside

    log['t_preprocessing'] = (time.process_time_ns() - start_process_time) * 1e-9
    log['wt_preprocessing'] = time.time() - start_wall_time

    print("Data preprocessing wall time: {}s".format(log['wt_preprocessing']))
    print("Data preprocessing process time: {}s".format(log['t_preprocessing']))

    #############################################
    start_wall_time = time.time()
    start_process_time = time.process_time_ns()

    # One view: DHF
    avg_dis = dhf_to_mesh_dis.mean(-1) + mesh_to_dhf_dis.mean(-1)
    if args.angle:
        avg_dis = avg_dis + np.minimum((avg_dis + 1e-4) * dhf_angle_scores.mean(-1), 1e-3)

    best_dhf_view = int(np.argmin(avg_dis))

    best_views = [best_dhf_view]
    min_cd = avg_dis[best_dhf_view]

    log['t_1'] = (time.process_time_ns() - start_process_time) * 1e-9
    log['wt_1'] = time.time() - start_wall_time
    log['cd_1'] = min_cd
    log['best_1'] = best_views

    print("Best DHF view:", best_dhf_view)
    print("Minimum DHF distances:", min_cd)
    print("View selection wall time (one DHF): {}s".format(log['wt_1']))
    print("View selection process time (one DHF): {}s".format(log['t_1']))

    os.makedirs(os.path.join(args.output_dir, "1"), exist_ok=True)
    np.savetxt(os.path.join(args.output_dir, "1", "view_directions.txt"), dhf_view_dirs[best_dhf_view][np.newaxis, :])
    if plane_z_origins is not None and plane_normals is not None:
        np.savetxt(os.path.join(args.output_dir, "1", "cutting_plane.txt"),
                   np.concatenate([plane_normals[0].squeeze(), np.array([plane_z_origins[0]])]))

    #############################################
    if args.angle:
        compute_chamfer_func = partial(compute_chamfer, mesh_to_dhf_dis=mesh_to_dhf_dis, mesh_to_hf_dis=mesh_to_hf_dis,
                                       dhf_floater_dis=dhf_floater_dis, hf_floater_dis=hf_floater_dis,
                                       dhf_floaters_to_hf_visibility=dhf_floaters_to_hf_visibility,
                                       hf_floaters_to_dhf_visibility=hf_floaters_to_dhf_visibility,
                                       hf_floaters_to_hf_visibility=hf_floaters_to_hf_visibility,
                                       dhf_angle_scores=dhf_angle_scores,
                                       hf_angle_scores=hf_angle_scores)
    else:
        compute_chamfer_func = partial(compute_chamfer, mesh_to_dhf_dis=mesh_to_dhf_dis, mesh_to_hf_dis=mesh_to_hf_dis,
                                       dhf_floater_dis=dhf_floater_dis, hf_floater_dis=hf_floater_dis,
                                       dhf_floaters_to_hf_visibility=dhf_floaters_to_hf_visibility,
                                       hf_floaters_to_dhf_visibility=hf_floaters_to_dhf_visibility,
                                       hf_floaters_to_hf_visibility=hf_floaters_to_hf_visibility)

    # More views: DHF + HF views
    for total_view_num in range(2, args.n_views + 1):
        if not args.no_skip:
            if min_cd < 1e-7:
                break

        start_wall_time = time.time()
        start_process_time = time.process_time_ns()

        n_samples = float(args.n_samples * total_view_num)
        best_view_candidates = []
        # Give a new tighter reference min_dist
        for i in range(args.n_hf):
            if i in best_views[1:]:
                continue
            cur_dis = compute_chamfer_func(best_views + [i], n_samples=n_samples)
            if min_cd > cur_dis:
                min_cd = cur_dis
                best_view_candidates = best_views + [i]

        if len(best_view_candidates) > 0:
            best_views = best_view_candidates

        print("Potential best views before search:", best_views)
        print("Current minimum Chamfer Distance:", min_cd)
        pruning_percentage = 1
        if min_cd > 0:
            all_hf_combinations = list(itertools.combinations(range(args.n_hf), total_view_num - 1))

            # init a thread-safe list to avoid thread contention
            manager = mp.Manager()
            shared_min_cd = manager.list([min_cd] * args.n_dhf)

            find_best_hfs_func = partial(find_best_hfs, all_hf_combinations=all_hf_combinations,
                                         shared_min_cd=shared_min_cd,
                                         dhf_floater_dis=dhf_floater_dis,
                                         hf_floater_dis=hf_floater_dis,
                                         dhf_floaters_to_hf_visibility=dhf_floaters_to_hf_visibility,
                                         hf_floaters_to_dhf_visibility=hf_floaters_to_dhf_visibility,
                                         hf_floaters_to_hf_visibility=hf_floaters_to_hf_visibility, n_samples=n_samples)

            if args.n_jobs > 1 and total_view_num > 2:
                results = Parallel(n_jobs=args.n_jobs)(delayed(find_best_hfs_func)(i, mesh_to_hf_dis=aux_mesh_dis_lst[i], angle_scores=aux_angle_scores_lst[i]) for i in tqdm(range(args.n_dhf)))
            else:
                results = [find_best_hfs_func(i, mesh_to_hf_dis=aux_mesh_dis_lst[i], angle_scores=aux_angle_scores_lst[i]) for i in tqdm(range(args.n_dhf))]

            total_pruning_count = 0
            for result in results:
                total_pruning_count += result[2]
                if min_cd > result[1]:
                    min_cd = result[1]
                    best_views = result[0]

            pruning_percentage = total_pruning_count / (len(all_hf_combinations) * args.n_dhf)

            log[str(total_view_num) + '_pruning'] = pruning_percentage
            log[str(total_view_num) + '_total_pruning_count'] = total_pruning_count
            log[str(total_view_num) + '_total_combinations'] = len(all_hf_combinations) * args.n_dhf


        if len(best_views) < total_view_num and not args.no_skip:
            print("Additional views cannot make the result better.")
            break

        while len(best_views) < total_view_num:
            print("Append arbitrary views to current {} views to compose {} views.".format(len(best_views), total_view_num))
            if args.no_skip:
                for i in range(args.n_hf):
                    if i not in best_views[1:]:
                        best_views += [i]
                        break

        # order the HF views by coverage:
        if total_view_num > 2:
            coverage_dict = dict()
            for i in best_views[1:]:
                # i * 1e-7 is an ugly trick to avoid exactly the same values
                coverage_dict[np.count_nonzero(mesh_to_hf_dis[i] > 0) / mesh_to_hf_dis[i].shape[0] + (i * 1e-7)] = i

            best_views = [best_views[0]] + [pair[1] for pair in sorted(coverage_dict.items())]

        log['t_' + str(total_view_num)] = (time.process_time_ns() - start_process_time) * 1e-9
        log['wt_' + str(total_view_num)] = time.time() - start_wall_time
        log['cd_' + str(total_view_num)] = min_cd
        log['best_' + str(total_view_num)] = best_views.copy()

        print("Best views:", best_views)
        print("Minimum Chamfer Distance:", min_cd)
        print("Process time: {}s".format(log['t_' + str(total_view_num)]))
        print("Wall time: {}s".format(log['wt_' + str(total_view_num)]))
        print("Pruning percentage is", pruning_percentage)
        print("*********************************************")

        dhf_view_dir = dhf_view_dirs[best_views[0]]
        hf_view_dir = hf_view_dirs[best_views[1:]]

        os.makedirs(os.path.join(args.output_dir, str(total_view_num)), exist_ok=True)
        np.savetxt(os.path.join(args.output_dir, str(total_view_num), "view_directions.txt"),
                   np.vstack([dhf_view_dir[np.newaxis, :], hf_view_dir]))
        if plane_z_origins is not None and plane_normals is not None:
            np.savetxt(os.path.join(args.output_dir, str(total_view_num), "cutting_plane.txt"),
                       np.concatenate([plane_normals[0].squeeze(), np.array([plane_z_origins[0]])]))

    with open(os.path.join(args.output_dir, "log.json"), 'w') as outfile:
        json.dump(log, outfile, indent=4)


if __name__ == '__main__':
    args = get_args()

    # create the output folder if not exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.using_axes:
        os.makedirs(os.path.join(args.output_dir, "5"), exist_ok=True)
        axes_view_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        np.savetxt(os.path.join(args.output_dir, "5", "view_directions.txt"),
                   axes_view_dirs)
    else:
        # if debug mode, create the output folder for visualization
        if args.dbg:
            os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)

        # load the mesh file
        mesh = trimesh.load(args.input_file, force='mesh')

        select_views(mesh)

