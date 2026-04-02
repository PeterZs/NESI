import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import trimesh
from pykdtree.kdtree import KDTree
import igl
import point_cloud_utils as pcu
import time



def normalize(v, dim=-1):
    """
    Normalize a tensor v to have unit norm along the specified dimension.
    :param v:
    :param dim:
    :return:
    """
    lib = np if isinstance(v, np.ndarray) else torch
    norm = lib.linalg.norm(v, axis=dim, keepdims=True)
    u = v / norm
    u[lib.isinf(u).any(-1)] = 0.0
    return u


def generate_random_dirs(n_samples, device=torch.device('cpu')):
    """
    Generate uniformly distributed random directions.
    :param n_samples: Number of random directions.
    :param device: 'cpu' or 'cuda'
    :return: Tensor of random directions, where each row is one random direction.
    """
    # the spatial normal distribution ensures the uniformness on the sphere
    dirs = torch.randn(n_samples, 3, device=device)
    dirs = normalize(dirs)
    return dirs


def align_vectors(u, v):
    """
    Compute rotation matrix from two unit vectors s.t. m @ u = v (u and v should be unit vectors).
    :param u: vector 1 (3-dimensional)
    :param v: vector 2 (3-dimensional)
    :return: rotation matrix
    """
    lib = np if isinstance(u, np.ndarray) else torch
    axis = lib.cross(u, v) if lib == np else lib.linalg.cross(u, v)

    if lib.linalg.norm(axis) < 1e-7:
        rot_mat = lib.eye(3)
        # when the angle is nearly 180 degree
        if (u * v).sum() < -0.99:
            rot_mat[0, 0] = -1
            rot_mat[2, 2] = -1

        if lib == torch:
            rot_mat = rot_mat.to(u.device)
        return rot_mat

    c = (u * v).sum()
    k = 1.0 / (1.0 + c)

    # ensure the output rot_mat is on the same device
    rot_mat = lib.stack([u, u, u])

    rot_mat[0, 0] = axis[0] * axis[0] * k + c
    rot_mat[0, 1] = axis[1] * axis[0] * k - axis[2]
    rot_mat[0, 2] = axis[2] * axis[0] * k + axis[1]

    rot_mat[1, 0] = axis[0] * axis[1] * k + axis[2]
    rot_mat[1, 1] = axis[1] * axis[1] * k + c
    rot_mat[1, 2] = axis[2] * axis[1] * k - axis[0]

    rot_mat[2, 0] = axis[0] * axis[2] * k - axis[1]
    rot_mat[2, 1] = axis[1] * axis[2] * k + axis[0]
    rot_mat[2, 2] = axis[2] * axis[2] * k + c

    return rot_mat


def spherical_fibonacci_sampling(n_samples):
    """
    Generate uniformly distributed points on a unit sphere by spherical Fibonacci sampling.
    Benjamin Keinert, Matthias Innmann, Michael Sänger, and Marc Stamminger.
    Spherical Fibonacci mapping. ACM Trans. Graph. 2015.
    https://doi.org/10.1145/2816795.2818131

    :param n_samples: the number of samples on the sphere
    :return: samples, shape is (n_samples, 3)
    """

    psi = np.sqrt(5.0) * 0.5 + 0.5

    point_ids = np.arange(0, n_samples)
    phi = 2.0 * np.pi * (point_ids / (psi - 1.0))
    cos_theta = 1.0 - (point_ids * 2.0 + 1.0) / float(n_samples)
    sin_theta_sqr = 1.0 - cos_theta * cos_theta

    sin_theta_sqr[sin_theta_sqr > 1.0] = 1.0

    #sin_theta = np.sqrt(np.max([0.0, np.min([sin_theta_sqr, 1.0])]))
    sin_theta = np.sqrt(sin_theta_sqr)

    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    z = cos_theta

    samples = np.stack([x, y, z], axis=-1)
    return samples


def colorize(scalar_array, min_value=None, max_value=None):
    import cv2
    scalar_array = scalar_array.squeeze()
    if isinstance(scalar_array, torch.Tensor):
        scalar_array = scalar_array.cpu().numpy()
    if min_value is None:
        min_value = np.min(scalar_array)
    if max_value is None:
        max_value = np.max(scalar_array)

    # generate a grayscale image
    scalar_array = (scalar_array - min_value) / (max_value - min_value + 1e-10) * 255
    scalar_array[scalar_array > 255] = 255

    # apply a colormap, then BGR->RGB
    return cv2.applyColorMap(scalar_array.astype(np.uint8), cv2.COLORMAP_JET).squeeze()[..., ::-1].copy()


def generate_view_directions(num_cameras, is_dhf, include_axes=True):
    """ wolrd_coors @ R + T -> NDC """

    if is_dhf:
        # Uniformly sample on the sphere
        camera_poses = spherical_fibonacci_sampling(num_cameras - 3)
        camera_poses = np.concatenate([camera_poses, np.eye(3)])
    else:
        # Uniformly sample on the sphere
        camera_poses = spherical_fibonacci_sampling(num_cameras - 6)
        camera_poses = np.concatenate([camera_poses, np.eye(3), -np.eye(3)])

    view_dirs = -camera_poses

    return view_dirs


def compute_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def ray_cast_by_embree(mesh, num_hits, num_rays_per_iter=10_000, verbose=False):
    # Count the number of hits
    cur_hits = 0
    if verbose:
        pbar = tqdm(total=num_hits)

    intersector = pcu.RayMeshIntersector(mesh.vertices, mesh.faces)

    hit_points = []


    while cur_hits < num_hits:

        # Sample uniformly inside the bounding box of [-1, 1]^3
        ray_origins = np.random.uniform(-1.0, 1.0, (num_rays_per_iter, 3))

        # Sample uniformly on a sphere
        ray_directions = normalize(np.random.randn(num_rays_per_iter, 3))

        fid, bc, ray_dis = intersector.intersect_rays(ray_origins, ray_directions)
        hit_mask = np.isfinite(ray_dis)

        # Set the hit point as the center point of the segment
        hit_points += [ray_origins[hit_mask] + ray_directions[hit_mask] * ray_dis[hit_mask][..., None]]
        cur_hits += np.count_nonzero(hit_mask)

        # To avoid infinite loop for a degenerated network
        if cur_hits == 0:
            return np.array([])

        if verbose:
            pbar.update(np.count_nonzero(hit_mask))

    if verbose:
        pbar.close()

    return np.concatenate(hit_points)


def fast_hf_sampling(esi, total_hits, num_rays_per_iter=1_000, num_samples_along_ray=200, num_iterations=3, verbose=False):

    # Count the number of hits
    hit_count = 0
    if verbose:
        pbar = tqdm(total=total_hits)

    # sampling on the ground truth mesh to compute CD
    mesh_samples, mesh_sample_fids = trimesh.sample.sample_surface(esi.mesh, total_hits)

    # potential robust issue for degenerated faces
    mesh_normals = esi.mesh.face_normals[mesh_sample_fids]

    # shift all sample points outside for checking visibility
    shifted_mesh_samples = mesh_samples + mesh_normals * 1e-5

    # check visibility
    visible_mask = esi.is_visible(shifted_mesh_samples)

    surface_samples = mesh_samples[visible_mask]

    hit_points = []

    while hit_count < total_hits - surface_samples.shape[0]:
        # Set ray origins as occluded sample points on the mesh
        ray_origins = mesh_samples[~visible_mask]

        # Sample uniformly on a sphere
        ray_directions = normalize(np.random.randn(ray_origins.shape[0], 3))

        ray_directions = ray_directions @ esi.r_mat[0]
        ray_directions[..., 2] = 0
        ray_directions = normalize(ray_directions)
        ray_directions = ray_directions @ esi.r_mat[0].T

        # The initial ray length
        ray_length = 3.5
        valid_num_rays = ray_origins.shape[0]

        # refining steps
        for j in range(num_iterations):
            # evenly sample points on the ray
            t = np.linspace(0, ray_length, num_samples_along_ray + 1)
            t = np.tile(t[None], (valid_num_rays, 1))
            sample_points = ray_origins[:, None, :] + ray_directions[:, None, :] * t[:, :, None]

            if j == 0:
                # filter out invalid points to save inference time
                bbox_mask = np.all(sample_points >= -1, axis=-1) & np.all(sample_points <= 1, axis=-1)

                # Forward the network
                inside_mask = esi.occupancy_check(sample_points[bbox_mask])

                is_inside = np.zeros(sample_points.shape[:-1], dtype=bool)
                is_inside[bbox_mask] = inside_mask

            else:
                # Forward the network
                is_inside = esi.occupancy_check(sample_points.reshape(-1, 3))
                is_inside = is_inside.reshape(sample_points.shape[:-1])

            # The intersection index i means the intersection point lies in [i, i+1]
            has_intersection = is_inside[..., :-1] != is_inside[..., 1:]

            # The mask if the ray has intersection
            ray_mask = np.any(has_intersection, axis=-1)

            sample_points = sample_points[ray_mask, ...]
            ray_directions = ray_directions[ray_mask, ...]
            has_intersection = has_intersection[ray_mask, ...]

            # Get the first intersection ID on each ray
            first_intersect_id = np.argmax(has_intersection.astype(np.float32), axis=1)

            # Shift the origin points of rays
            ray_origins = sample_points[np.arange(sample_points.shape[0]), first_intersect_id]

            # filter out the points outside the bounding box ([-1, 1])
            bbox_filter_mask = np.all(ray_origins >= -1, axis=-1) & np.all(ray_origins <= 1, axis=-1)
            ray_origins = ray_origins[bbox_filter_mask]
            ray_directions = ray_directions[bbox_filter_mask]

            ray_length = ray_length / num_samples_along_ray

            valid_num_rays = ray_origins.shape[0]
            if valid_num_rays == 0:
                break

        # Set the hit point as the center point of the segment
        hit_points += [ray_origins + ray_directions * ray_length * 0.5]
        hit_count += ray_origins.shape[0]

        # To avoid infinite loop for a degenerated network
        if hit_count == 0:
            return np.array([])

        if verbose:
            pbar.update(ray_origins.shape[0])

    if verbose:
        pbar.close()

    return np.concatenate([surface_samples] + hit_points)[:total_hits]


def sample_by_ray_casting(occ_func, total_hits, num_rays_per_iter=1_000, num_samples_along_ray=200, num_iterations=3, ray_length=3.5, device=None, verbose=False):
    # Check the device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Count the number of hits
    hit_count = 0
    if verbose:
        pbar = tqdm(total=total_hits)

    hit_points = []

    while hit_count < total_hits:
        # Sample uniformly inside the bounding box of [-1, 1]^3
        ray_origins = torch.FloatTensor(num_rays_per_iter, 3).uniform_(-1, 1).to(device)

        # Sample uniformly on a sphere
        ray_directions = generate_random_dirs(num_rays_per_iter, device)

        # The initial ray length
        ray_l = ray_length
        valid_num_rays = num_rays_per_iter

        # refining steps
        for j in range(num_iterations):
            # evenly sample points on the ray
            t = torch.linspace(0, ray_l, num_samples_along_ray + 1, device=device)
            t = t[None].expand(valid_num_rays, -1)
            sample_points = ray_origins[:, None, :] + ray_directions[:, None, :] * t[:, :, None]

            if j == 0:
                # filter out invalid points to save inference time
                bbox_mask = (sample_points >= -1).all(dim=-1) & (sample_points <= 1).all(dim=-1)

                # Forward the network
                inside_mask = occ_func(sample_points[bbox_mask])

                is_inside = torch.zeros(sample_points.shape[:-1], dtype=torch.bool, device=device)
                is_inside[bbox_mask] = inside_mask

            else:
                # Forward the network
                is_inside = occ_func(sample_points.reshape(-1, 3))
                is_inside = is_inside.reshape(sample_points.shape[:-1])

            # The intersection index i means the intersection point lies in [i, i+1]
            has_intersection = is_inside[..., :-1] != is_inside[..., 1:]

            # The mask if the ray has intersection
            ray_mask = has_intersection.any(dim=-1)

            sample_points = sample_points[ray_mask, ...]
            ray_directions = ray_directions[ray_mask, ...]
            has_intersection = has_intersection[ray_mask, ...]

            # Get the first intersection ID on each ray
            first_intersect_id = torch.argmax(has_intersection.float(), dim=1)

            # Shift the origin points of rays
            ray_origins = sample_points[torch.arange(sample_points.shape[0]), first_intersect_id]

            # filter out the points outside the bounding box ([-1, 1])
            bbox_filter_mask = (ray_origins >= -1).all(dim=-1) & (ray_origins <= 1).all(dim=-1)
            ray_origins = ray_origins[bbox_filter_mask]
            ray_directions = ray_directions[bbox_filter_mask]

            ray_l = ray_l / num_samples_along_ray

            valid_num_rays = ray_origins.shape[0]
            if valid_num_rays == 0:
                break

        # Set the hit point as the center point of the segment
        hit_points += [ray_origins + ray_directions * ray_l * 0.5]
        hit_count += ray_origins.shape[0]

        # To avoid infinite loop for a degenerated network
        if hit_count == 0:
            return np.array([])

        if verbose:
            pbar.update(ray_origins.shape[0])

    if verbose:
        pbar.close()

    return torch.cat(hit_points)[:total_hits]


def esi_ray_cast_fast(esi, total_hits, num_rays_per_iter=1_000, num_samples_along_ray=200, num_iterations=3, verbose=False):

    # Count the number of hits
    hit_count = 0
    if verbose:
        pbar = tqdm(total=total_hits)

    hit_points = []

    while hit_count < total_hits:
        # Sample uniformly inside the bounding box of [-1, 1]^3
        ray_origins = np.random.uniform(-1.0, 1.0, (num_rays_per_iter, 3))

        # Sample uniformly on a sphere
        ray_directions = normalize(np.random.randn(num_rays_per_iter, 3))

        # check if the ray shoots to infinity by using the convex hull
        _, _, dis_to_o = esi.ch_intersector.intersect_rays(ray_origins, ray_directions)
        ray_origins = ray_origins[np.isfinite(dis_to_o)]
        ray_directions = ray_directions[np.isfinite(dis_to_o)]

        # fid is the index of each face intersected (-1 for ray miss)
        # bc are the barycentric coordinates of each intersected ray
        # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
        fid, bc, dis_to_o = esi.intersector.intersect_rays(ray_origins, ray_directions)
        finite_mask = np.isfinite(dis_to_o)
        intersection_points = ray_origins[finite_mask] + ray_directions[finite_mask] * dis_to_o[finite_mask][..., np.newaxis]

        # check the surface visibility
        shifted_points = intersection_points + esi.mesh.face_normals[fid[finite_mask]] * 1e-5
        visible_mask = esi.is_visible(shifted_points)

        # check the visibility along the entire ray
        shifted_points = intersection_points[visible_mask] - ray_directions[finite_mask][visible_mask] * 1e-5
        hits = esi.ray_cast(ray_origins[finite_mask][visible_mask], shifted_points - ray_origins[finite_mask][visible_mask], ray_length=1.0)

        # combine both visibility
        visible_mask[visible_mask.copy()] = np.isinf(hits).all(-1)

        # set the valid intersection points as real hits
        hit_points += [intersection_points[visible_mask]]
        hit_count += np.count_nonzero(visible_mask)
        if verbose:
            pbar.update(np.count_nonzero(visible_mask))

        valid_mask = np.ones(ray_origins.shape[0], dtype=bool)
        valid_mask[finite_mask] = ~visible_mask

        ray_origins = ray_origins[valid_mask]
        ray_directions = ray_directions[valid_mask]

        hits = esi.ray_cast(ray_origins, ray_directions, ray_length=3.5)

        # # Set the hit point as the center point of the segment
        hit_mask = np.isfinite(hits).all(-1)
        hit_points += [hits[hit_mask]]
        hit_count += np.count_nonzero(hit_mask)

        # To avoid infinite loop for a degenerated network
        if hit_count == 0:
            return np.array([])

        if verbose:
            pbar.update(np.count_nonzero(hit_mask))

    if verbose:
        pbar.close()

    return np.concatenate(hit_points)[:total_hits]



def is_visible(view_dir, points, intersector, is_dhf=True, z_origin=-2):
    """
    Check if points are visible from the specific view direction
    :param view_dir:
    :param points:
    :param intersector:
    :param is_dhf:
    :return: The visibility mask of the input points
    """
    points = points.astype(np.float64)
    batch_view_dir = np.tile(view_dir[None, ...], (points.shape[0], 1))
    _, _, front_dis = intersector.intersect_rays(points, -batch_view_dir)

    # compute the signed distance to the plane for points
    src_signed_dis = np.dot(points, view_dir) - z_origin

    # compute the signed distance to the plane for intersection points
    intersection_points = points - batch_view_dir * front_dis[..., np.newaxis]
    tgt_signed_dis = np.dot(intersection_points, view_dir) - z_origin
    tgt_signed_dis[np.isinf(front_dis)] = np.inf

    visible_mask = (src_signed_dis >= 0) & ((tgt_signed_dis <= 0) | np.isinf(front_dis))

    if is_dhf:
        _, _, back_dis = intersector.intersect_rays(points, batch_view_dir)
        visible_mask = visible_mask | np.isinf(back_dis)

    return visible_mask


def create_arrow(direction=np.array([0., 0., 1.]), arrow_dis=1.5):
    """

    Create an arrow mesh aligning with the input direction
    :param direction: The direction of the arrow
    :param arrow_dis: the distance from the arrow head to the origin
    :return: An arrow mesh aligning with the input direction
    """
    arrow_head = trimesh.creation.cone(radius=0.05, height=0.2)

    # mirror the arrow head
    arrow_head.vertices[:, 2] *= -1
    arrow_head.faces = np.fliplr(arrow_head.faces)
    arrow_head.apply_translation([0, 0, -(arrow_dis+0.2)])

    arrow_body = trimesh.creation.cylinder(radius=0.025, height=0.4)
    arrow_body.apply_translation([0, 0, -arrow_dis])

    arrow = trimesh.util.concatenate(arrow_head, arrow_body)

    r_mat = np.eye(4)
    r_mat[:3, :3] = align_vectors(np.array([0., 0., 1.]), direction)
    arrow.apply_transform(r_mat)

    return arrow


def export_points_with_values(points, values, filename):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    try:
        # for the new version of pymeshlab
        m = pymeshlab.Mesh(points, v_scalar_array=values)
    except:
        # for the version before 2021.7
        m = pymeshlab.Mesh(points, v_quality_array=values)
    ms.add_mesh(m, "cur_mesh")
    ms.save_current_mesh(filename)


def compute_distances(pred_v, gt_v, verbose_return=False):
    """
    Compute distances between two point clouds
    :param pred_v:
    :param gt_v:
    :param verbose_return:
    :return:
    """

    metrics = {}

    # from pred to gt
    gt_tree = KDTree(gt_v.astype(np.float64))
    pred_to_gt_dist, inds = gt_tree.query(pred_v.astype(np.float64), k=1)

    pred_tree = KDTree(pred_v.astype(np.float64))
    gt_to_pred_dist, inds = pred_tree.query(gt_v.astype(np.float64), k=1)

    metrics['mean_0'] = pred_to_gt_dist.mean()
    metrics['max_0'] = pred_to_gt_dist.max()
    metrics['min_0'] = pred_to_gt_dist.min()
    metrics['L2_0'] = (pred_to_gt_dist ** 2).mean()
    metrics['RMS_0'] = np.sqrt(metrics['L2_0'])
    metrics['diag_mesh_0'] = np.linalg.norm(np.ptp(pred_v, 0))

    metrics['mean_1'] = gt_to_pred_dist.mean()
    metrics['max_1'] = gt_to_pred_dist.max()
    metrics['min_1'] = gt_to_pred_dist.min()
    metrics['L2_1'] = (gt_to_pred_dist ** 2).mean()
    metrics['RMS_1'] = np.sqrt(metrics['L2_1'])
    metrics['diag_mesh_1'] = np.linalg.norm(np.ptp(gt_v, 0))

    metrics['Chamfer'] = (metrics['mean_0'] / metrics['diag_mesh_1'] + metrics['mean_1'] / metrics['diag_mesh_1']) / 2.0
    metrics['L2'] = (metrics['L2_0'] / metrics['diag_mesh_1'] + metrics['L2_1'] / metrics['diag_mesh_1']) / 2.0
    metrics['RMS'] = (metrics['RMS_0'] / metrics['diag_mesh_1'] + metrics['RMS_1'] / metrics['diag_mesh_1']) / 2.0

    float_metrics = {k: float(v) for k, v in metrics.items()}

    if verbose_return:
        return float_metrics, pred_to_gt_dist
    else:
        return float_metrics


def point_mesh_distances(pts, mesh):
    dist, _, _ = igl.point_mesh_squared_distance(pts.reshape(-1, 3), mesh.vertices, mesh.faces)
    return np.sqrt(dist)


def ray_aabb_intersection(ray_o, ray_d, aabb_min, aabb_max):
    """
    http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    """
    dir_fraction = np.divide(1.0, ray_d)

    t_min = (aabb_min - ray_o) * dir_fraction
    t_max = (aabb_max - ray_o) * dir_fraction

    # the infinities are used here to eliminate NaNs that
    # are generated when the ray sits on a boundary plane
    tmin = np.minimum(t_min, t_max).max(-1)
    tmax = np.maximum(t_min, t_max).min(-1)

    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    tmin[tmax < 0] = np.inf
    tmin[tmin > tmax] = np.inf

    # t is the distance from the ray point
    # to intersection
    point = ray_o + (ray_d * tmin[..., None])
    return point


def training_data_sampling(view_dir, mesh, z_origin=-2.0, n_samples=5_000_000, is_dhf=False, prev_view_dirs=None, with_mask=False):
    # Compute the rotation matrix, which provides the local frame
    z_axis = np.array([0, 0, 1])
    r_mat = align_vectors(view_dir, z_axis).T

    mesh_intersector = pcu.RayMeshIntersector(mesh.vertices, mesh.faces)

    # sampling on surface
    surface_samples_wc, surface_sample_fid = mesh.sample(n_samples, return_index=True)
    surface_normals_wc = mesh.face_normals[surface_sample_fid]

    # Detect non-grazing regions by a threshold 89.9
    non_grazing_mask = (surface_normals_wc * -view_dir).sum(-1) > np.cos(88. * np.pi / 180.)

    # Shift out a little bit for visibility check
    shifted_surface_samples = surface_samples_wc + surface_normals_wc * 1e-5

    # Broadcast view directions to be the same shape as samples
    shifted_surface_samples, batch_view_dir = np.broadcast_arrays(shifted_surface_samples, view_dir)

    # fid is the index of each face intersected (-1 for ray miss)
    # bc are the barycentric coordinates of each intersected ray
    # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
    visible_mask = is_visible(view_dir, shifted_surface_samples, mesh_intersector, is_dhf=False, z_origin=z_origin)

    # The sample points and their normals on the surface
    surface_samples_lc = surface_samples_wc[visible_mask & non_grazing_mask] @ r_mat
    surface_normals_lc = surface_normals_wc[visible_mask & non_grazing_mask] @ r_mat

    # uniformly sample in [-1.01, 1.01]^3 and rotate the sample points into local coordinates
    uni_samples_lc = np.random.uniform(-1.01, 1.01, (n_samples, 3)) @ r_mat
    # remove the depth
    uni_samples_lc[:, 2] = z_origin
    # rotate them back the world coordinates
    uni_samples_wc = uni_samples_lc @ r_mat.T

    uni_samples_wc, batch_view_dir = np.broadcast_arrays(uni_samples_wc, view_dir)
    # fid is the index of each face intersected (-1 for ray miss)
    # bc are the barycentric coordinates of each intersected ray
    # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
    fid, _, ray_dis = mesh_intersector.intersect_rays(uni_samples_wc, batch_view_dir.copy())
    spatial_mask = np.isinf(ray_dis)
    surface_mask = np.isfinite(ray_dis)

    # keep only non-grazing surface samples
    surface_mask[surface_mask.copy()] = (mesh.face_normals[fid[surface_mask]] * -view_dir).sum(-1) > np.cos(85. * np.pi / 180.)

    # Generate surface samples
    uni_samples_lc[..., 2] = ray_dis + z_origin

    surface_samples_lc = np.vstack([surface_samples_lc, uni_samples_lc[surface_mask]])
    surface_normals_lc = np.vstack([surface_normals_lc, mesh.face_normals[fid[surface_mask]] @ r_mat])

    spatial_samples_lc = uni_samples_lc[spatial_mask]

    # cutting plane
    if z_origin > -1.9:
        # inner_mask = (surface_normals_wc * -view_dir).sum(-1) < -np.cos(89.9 * np.pi / 180.)
        inner_mask = np.isfinite(ray_dis)
        inner_mask[inner_mask.copy()] = (mesh.face_normals[fid[inner_mask]] * -view_dir).sum(-1) < -np.cos(
            89.9 * np.pi / 180.)
        inner_samples_lc = uni_samples_lc[inner_mask]
        inner_samples_lc[:, 2] = z_origin
        inner_normals_lc = mesh.face_normals[fid[inner_mask]] @ r_mat

        n_valid_surface_samples = surface_samples_lc.shape[0]
        surface_samples_lc = np.vstack([surface_samples_lc, inner_samples_lc])
        surface_normals_lc = np.vstack([surface_normals_lc, inner_normals_lc])

        roi_mask = np.zeros(surface_samples_lc.shape[0], dtype=bool)
        roi_mask[:n_valid_surface_samples] = True
        expanded_roi_mask = roi_mask.copy()
    else:
        roi_mask = None
        expanded_roi_mask = None

    if is_dhf:
        # reuse the surface sampling
        shifted_surface_samples, batch_view_dir = np.broadcast_arrays(shifted_surface_samples, view_dir)

        _, _, ray_dis = mesh_intersector.intersect_rays(shifted_surface_samples, batch_view_dir.copy())
        visible_mask = np.isinf(ray_dis)
        non_grazing_mask = (surface_normals_wc * view_dir).sum(-1) > np.cos(88. * np.pi / 180.)

        back_surface_samples_lc = surface_samples_wc[visible_mask & non_grazing_mask] @ r_mat
        back_surface_normals_lc = surface_normals_wc[visible_mask & non_grazing_mask] @ r_mat

        # reuse the spatial sampling
        # remove the depth
        uni_samples_lc[:, 2] = -z_origin
        # rotate them back the world coordinates
        uni_samples_wc = uni_samples_lc @ r_mat.T

        uni_samples_wc, batch_view_dir = np.broadcast_arrays(uni_samples_wc, view_dir)
        # fid is the index of each face intersected (-1 for ray miss)
        # bc are the barycentric coordinates of each intersected ray
        # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
        fid, _, ray_dis = mesh_intersector.intersect_rays(uni_samples_wc, -batch_view_dir.copy())
        spatial_mask = np.isinf(ray_dis)
        surface_mask = np.isfinite(ray_dis)

        # keep only non-grazing surface samples
        surface_mask[surface_mask.copy()] = np.abs(
            (mesh.face_normals[fid[surface_mask]] * view_dir).sum(-1)) > np.cos(85. * np.pi / 180.)

        # Generate surface samples
        uni_samples_lc[..., 2] = -ray_dis + abs(z_origin)

        back_surface_samples_lc = np.vstack([back_surface_samples_lc, uni_samples_lc[surface_mask]])
        back_surface_normals_lc = np.vstack([back_surface_normals_lc, mesh.face_normals[fid[surface_mask]] @ r_mat])

        # Using nan to mark missing data
        front_surface_samples_lc = np.hstack([surface_samples_lc, np.full((surface_samples_lc.shape[0], 1), np.nan)])
        back_surface_samples_lc = np.hstack([back_surface_samples_lc[:, :2], np.full((back_surface_samples_lc.shape[0], 1), np.nan), back_surface_samples_lc[:, -1:]])
        surface_samples_lc = np.vstack([front_surface_samples_lc, back_surface_samples_lc])

        front_normals_lc = np.stack([surface_normals_lc, surface_normals_lc * np.nan])
        back_normals_lc =  np.stack([back_surface_normals_lc * np.nan, back_surface_normals_lc])
        surface_normals_lc = np.concatenate([front_normals_lc, back_normals_lc], axis=1)

    else:
        if with_mask:
            from models.esi import ESI

            # Compose ESI using previous view directions
            prev_esi = ESI(prev_view_dirs, mesh, dhf_ids=[0])

            # Compute the intersection points between rays and the previous ESI
            surface_samples_wc = surface_samples_lc @ r_mat.T
            surface_normals_wc = surface_normals_lc @ r_mat.T

            shifted_samples = surface_samples_wc + surface_normals_wc * 1e-5

            # check if the points are visible from the previous ESI
            visible_mask = prev_esi.is_visible(shifted_samples)

            # Set ray origins, only check those visible points
            ray_origin_lc = surface_samples_lc[visible_mask]
            ray_origin_lc[..., 2] = z_origin
            ray_origin_wc = ray_origin_lc @ r_mat.T

            # Set the view direction
            view_dirs = ray_origin_wc - shifted_samples[visible_mask]
            esi_intersection_points = prev_esi.batch_ray_cast(shifted_samples[visible_mask], view_dirs, ray_length=1.0)

            # If the ray hits infinity, then the whole ray is visible (not intersect to the previous ESI); otherwise it is not visible
            visible_mask[visible_mask.copy()] = np.isinf(esi_intersection_points).any(-1)#dis < 1e-4

            # check if the points are well visible from previous ESI
            view_score = prev_esi.get_visible_score(surface_samples_wc[visible_mask], surface_normals_wc[visible_mask])
            # a point is well visible if it is not in the grazing region of ESI
            well_visible_mask = view_score > np.cos(70.0 * np.pi / 180.0)

            # and if current view score is not better than the previous ones in grazing region
            cur_view_score = (surface_normals_lc[visible_mask] * np.array([0, 0, -1])).sum(-1)
            well_visible_mask = np.bitwise_or(well_visible_mask, np.bitwise_and(view_score <= np.cos(70.0 * np.pi / 180.0), cur_view_score < view_score))

            visible_mask[visible_mask.copy()] = well_visible_mask
            roi_mask = ~visible_mask
            roi_samples = surface_samples_lc[roi_mask]

            # Dilate the supervised region
            gt_tree = KDTree(roi_samples)
            dis, _ = gt_tree.query(surface_samples_lc, k=1)
            roi_mask = (dis < 5e-3).squeeze()
            expanded_roi_mask = (dis < 1e-2).squeeze()

            # protection
            normal_visibility = (surface_normals_lc * np.array([0, 0, -1])).sum(-1) > 0

            roi_mask = roi_mask & normal_visibility
            expanded_roi_mask = expanded_roi_mask & normal_visibility

    return r_mat, surface_samples_lc, surface_normals_lc, spatial_samples_lc, roi_mask, expanded_roi_mask


def normalized_grid(width, height):
    """Returns grid[x,y] -> coordinates for a normalized window.

    Args:
        width, height (int): grid resolution
    """

    # These are normalized coordinates
    # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
    window_x = np.linspace(1, -1, num=width) * (width / height)
    window_x += np.random.rand(*window_x.shape) * (1. / width)
    window_y = np.linspace(1, -1, num=height)
    window_y += np.random.rand(*window_y.shape) * (1. / height)
    coord = np.array(np.meshgrid(window_x, window_y, indexing='xy')).transpose(1, 2, 0)

    return coord



def generate_2d_omega_mask(view_dir, mesh, width, height, z_origin=-2.0, is_dhf=False, prev_view_dirs=None, with_mask=False):
    # Compute the rotation matrix, which provides the local frame
    z_axis = np.array([0, 0, 1])
    r_mat = align_vectors(view_dir, z_axis).T

    mesh_intersector = pcu.RayMeshIntersector(mesh.vertices, mesh.faces)

    coord = normalized_grid(width, height)

    # uniformly sample in [-1.01, 1.01]^3 and rotate the sample points into local coordinates
    uni_samples_lc = np.zeros([width * height, 3])
    uni_samples_lc[:, :2] = coord.reshape(-1, 2)
    # remove the depth
    uni_samples_lc[:, 2] = z_origin
    # rotate them back the world coordinates
    uni_samples_wc = uni_samples_lc @ r_mat.T

    uni_samples_wc, batch_view_dir = np.broadcast_arrays(uni_samples_wc, view_dir)
    # fid is the index of each face intersected (-1 for ray miss)
    # bc are the barycentric coordinates of each intersected ray
    # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
    fid, _, ray_dis = mesh_intersector.intersect_rays(uni_samples_wc, batch_view_dir.copy())
    spatial_mask = np.isinf(ray_dis)
    surface_mask = np.isfinite(ray_dis)

    # keep only non-grazing surface samples
    surface_mask[surface_mask.copy()] = (mesh.face_normals[fid[surface_mask]] * -view_dir).sum(-1) > np.cos(85. * np.pi / 180.)

    # Generate surface samples
    uni_samples_lc[..., 2] = ray_dis + z_origin

    surface_samples_lc = uni_samples_lc[surface_mask]
    surface_normals_lc = mesh.face_normals[fid[surface_mask]] @ r_mat
    surface_v_color = get_texture(mesh, surface_samples_lc @ r_mat.T, fid[surface_mask])

    texture_image = np.ones([width * height, 3])
    texture_image[surface_mask] = surface_v_color

    if not is_dhf:
        if with_mask:
            from models.esi import ESI

            # Compose ESI using previous view directions
            prev_esi = ESI(prev_view_dirs, mesh, dhf_ids=[0])

            # Compute the intersection points between rays and the previous ESI
            surface_samples_wc = surface_samples_lc @ r_mat.T
            surface_normals_wc = surface_normals_lc @ r_mat.T

            shifted_samples = surface_samples_wc + surface_normals_wc * 1e-5

            # check if the points are visible from the previous ESI
            visible_mask = prev_esi.is_visible(shifted_samples)

            # Set ray origins, only check those visible points
            ray_origin_lc = surface_samples_lc[visible_mask]
            ray_origin_lc[..., 2] = z_origin
            ray_origin_wc = ray_origin_lc @ r_mat.T

            # Set the view direction
            view_dirs = ray_origin_wc - shifted_samples[visible_mask]
            esi_intersection_points = prev_esi.batch_ray_cast(shifted_samples[visible_mask], view_dirs, ray_length=1.0)

            # If the ray hits infinity, then the whole ray is visible (not intersect to the previous ESI); otherwise it is not visible
            visible_mask[visible_mask.copy()] = np.isinf(esi_intersection_points).any(-1)#dis < 1e-4

            # An alternative method, but seems not so fast
            # ray_origin_wc, view_dirs = np.broadcast_arrays(ray_origin_wc, view_dir)
            # esi_intersection_points = prev_esi.intersect_rays(ray_origin_wc, view_dirs, ray_length=4.0)
            # dis = np.linalg.norm(esi_intersection_points - surface_samples_wc, axis=-1)
            # visible_mask[visible_mask.copy()] = dis < 1e-4

            # check if the points are well visible from previous ESI
            view_score = prev_esi.get_visible_score(surface_samples_wc[visible_mask], surface_normals_wc[visible_mask])
            # a point is well visible if it is not in the grazing region of ESI
            well_visible_mask = view_score > np.cos(70.0 * np.pi / 180.0)

            # and if current view score is not better than the previous ones in grazing region
            cur_view_score = (surface_normals_lc[visible_mask] * np.array([0, 0, -1])).sum(-1)
            well_visible_mask = np.bitwise_or(well_visible_mask, np.bitwise_and(view_score <= np.cos(70.0 * np.pi / 180.0), cur_view_score < view_score))

            visible_mask[visible_mask.copy()] = well_visible_mask
            roi_mask = ~visible_mask
            roi_samples = surface_samples_lc[roi_mask]

            # Dilate the supervised region
            gt_tree = KDTree(roi_samples)
            dis, _ = gt_tree.query(surface_samples_lc, k=1)
            roi_mask = (dis < 5e-3).squeeze()
            expanded_roi_mask = (dis < 1e-2).squeeze()

            # protection
            normal_visibility = (surface_normals_lc * np.array([0, 0, -1])).sum(-1) > 0

            roi_mask = roi_mask & normal_visibility
            expanded_roi_mask = expanded_roi_mask & normal_visibility

            surface_mask[surface_mask.copy()] = expanded_roi_mask

    texture_image[~surface_mask] = 1

    return surface_mask.reshape(width, height), texture_image.reshape(width, height, 3)



def image2mesh(hf_image, min_corner=(-1, -1), max_corner=(1, 1)):
    h, w = hf_image.squeeze().shape
    faces = np.empty((h-1, w-1, 2, 3))
    vid_array = np.linspace(0, h*w, num=h*w, endpoint=False, dtype=int).reshape(h, w)
    faces[:, :, 0, 0] = vid_array[:-1, :-1]
    faces[:, :, 0, 1] = vid_array[1:, 1:]
    faces[:, :, 0, 2] = vid_array[:-1, 1:]


    faces[:, :, 1, 0] = vid_array[1:, 1:]
    faces[:, :, 1, 1] = vid_array[:-1, :-1]
    faces[:, :, 1, 2] = vid_array[1:, :-1]

    faces = faces.reshape(-1, 3)

    xs = np.linspace(max_corner[0] - (max_corner[0] - min_corner[0]) * 0.5 / w, min_corner[0] + (max_corner[0] - min_corner[0]) * 0.5 / w, num=w)
    ys = np.linspace(max_corner[1] - (max_corner[1] - min_corner[1]) * 0.5 / h, min_corner[1] + (max_corner[1] - min_corner[1]) * 0.5 / h, num=h)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    vertices = np.stack([x.reshape(-1), y.reshape(-1), hf_image.reshape(-1)], axis=-1)

    v_valid_mask = np.isfinite(vertices).all(-1)
    f_valid_mask = v_valid_mask[faces.astype(int)].all(-1)
    faces = faces[f_valid_mask]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.remove_unreferenced_vertices()

    return mesh


def eval_views(net, device, res=1000, compute_normal=True):
    view_range = [-1.5, 1.5]
    pixel_length = (view_range[1] - view_range[0]) / res

    xs = torch.linspace(view_range[1] - pixel_length * 0.5, view_range[0] + pixel_length * 0.5, steps=res)
    ys = torch.linspace(view_range[1] - pixel_length * 0.5, view_range[0] + pixel_length * 0.5, steps=res)
    x, y = torch.meshgrid(xs, ys, indexing='xy')

    meshes = []

    for view_i in range(net.get_n_views()):
        points = torch.stack([x, y], axis=-1).reshape(-1, 2).to(device)
        points = points.requires_grad_(True)

        cur_dhf = net(view_i, points)

        # # if the last channel is mask
        if view_i > 0:

            cur_mask = nn.Sigmoid()(cur_dhf[..., -1].detach()) > 0.5#nn.Sigmoid()(cur_dhf[..., -1].detach()).cpu().numpy()
            cur_dhf = cur_dhf[..., :-1]
            # #########################

        for k in range(cur_dhf.shape[-1]):
            cur_hf = cur_dhf[..., k]

            if compute_normal:
                # Compute the gradient of HF
                hf_gradients = compute_gradient(cur_hf, points).detach().cpu()

                # Compose normals
                normals = torch.concat([-hf_gradients, torch.ones(hf_gradients.shape[0])[..., None]], dim=-1).numpy()
                normals = normalize(normals)

            if cur_dhf.shape[-1] >= 2:
                opposite_hf = cur_dhf[..., (k + 1) % 2]
                if k == 0:
                    #cur_hf[cur_hf > opposite_hf] = 4
                    valid_v_mask = (cur_hf < opposite_hf).cpu().numpy()
                else:
                    #cur_hf[cur_hf < opposite_hf] = 1e-3
                    valid_v_mask = (cur_hf > opposite_hf).cpu().numpy()

            cur_hf = cur_hf.detach().reshape(res, res)

            mesh = image2mesh(cur_hf.cpu().numpy(), min_corner=(view_range[0], view_range[0]), max_corner=(view_range[1], view_range[1]))

            if k == 1:
                # flip the face orientation
                mesh.faces = np.fliplr(mesh.faces)

            # convert vertices from NDC to the world coordinates
            vertices_ndc = mesh.vertices.copy()
            mesh.vertices = mesh.vertices @ net.r_mat[view_i].cpu().numpy().T
            if compute_normal:
                mesh.vertex_normals = -normals @ net.r_mat[view_i].cpu().numpy().T
                mesh.visual.vertex_colors = mesh.vertex_normals * 0.5 + 0.5

            # crop vertices using the bounding box in the world coordinates
            if view_i > 0:

                mesh.faces = np.fliplr(mesh.faces)

                valid_v_mask = cur_mask.cpu().numpy() & (mesh.vertices > -1).all(axis=-1) & (mesh.vertices < 1).all(axis=-1)
                vertices_ndc[~valid_v_mask, 2] = 4.

            else:
                valid_v_mask = valid_v_mask & (mesh.vertices > -1).all(axis=-1) & (mesh.vertices < 1).all(axis=-1)

            # get faces in which all vertices are valid
            valid_f_mask = valid_v_mask[mesh.faces].all(axis=-1)  # & f_normal_mask

            # filter out invalid faces and vertices
            mesh.update_faces(valid_f_mask)
            mesh.remove_unreferenced_vertices()

            meshes += [mesh]


    overall_mesh = trimesh.util.concatenate([meshes[0], meshes[1]])

    new_meshes = [overall_mesh] + meshes[2:]

    return new_meshes


def calculate_normals_by_finite_diff(x):

    middle = x[1:-1, 1:-1, :]
    up = x[1:-1, 2:, :]
    down = x[1:-1, :-2, :]
    left = x[:-2, 1:-1, :]
    right = x[2:, 1:-1, :]
    left_up = x[:-2, 2:, :]
    right_up = x[2:, 2:, :]
    left_down = x[:-2, :-2, :]
    right_down = x[2:, :-2, :]

    normal_1 = get_v_normal(middle, left, up, right, down)
    normal_2 = get_v_normal2(middle, left_up, right_up, left_down, right_down)
    normal = (normal_1 + normal_2)/2

    normal = normal / torch.norm(normal, dim=-1, keepdim=True)

    # pad with 0s
    normal = torch.cat([torch.zeros((1, normal.shape[1], 3)).to(normal.device),
                       normal, torch.zeros((1, normal.shape[1], 3)).to(normal.device)], dim=0)
    normal = torch.cat([torch.zeros((normal.shape[0], 1, 3)).to(normal.device),
                       normal, torch.zeros((normal.shape[0], 1, 3)).to(normal.device)], dim=1)

    return normal.reshape(x.shape)


def get_v_normal(v, left, up, right, down):

    normal1 = get_tri_normal(v, left, up)
    normal2 = get_tri_normal(v, up, right)
    normal3 = get_tri_normal(v, right, down)
    normal4 = get_tri_normal(v, down, left)

    normal = (normal1 + normal2 + normal3 + normal4)/4

    return normal


def get_v_normal2(v, up_left, up_right, down_left, down_right):

    normal1 = get_tri_normal(v, up_left, up_right)
    normal2 = get_tri_normal(v, up_right, down_right)
    normal3 = get_tri_normal(v, down_right, down_left)
    normal4 = get_tri_normal(v, down_left, up_left)

    normal = (normal1 + normal2 + normal3 + normal4)/4

    return normal


def get_tri_normal(v0, v1, v2):
    e1 = v1-v0
    e2 = v2-v0

    normal = torch.linalg.cross(e1, e2)
    # normal = normal / (normal * normal)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)

    return normal


def update_vertices_by_shooting_rays(mesh, net, v_mask, view_R, check_range, device):
    ray_dir = torch.Tensor(mesh.vertex_normals[v_mask])

    # project the normal direction to the DHF view plane to ensure no face flipping after offsets
    ray_dir = ray_dir @ view_R.cpu().numpy()
    ray_dir[..., 2] = 0
    ray_dir = ray_dir @ view_R.cpu().numpy().T

    #pixel_length = (view_range[1] - view_range[0]) / res
    ori_points = torch.Tensor(mesh.vertices[v_mask])# - ray_dir * 1e-4#check_range

    chunks = ori_points.shape[0] // 5_000 + 1
    net.batch_ray_cast(ori_points, ray_dir, max_rays_per_chunk=10_000, ray_length=check_range * 1.8,
                       n_samples_along_ray=200, n_iter=1, device=None)
    new_points = net.batch_ray_cast(ori_points, ray_dir, max_rays_per_chunk=10_000, ray_length=check_range * 1.8,
                       n_samples_along_ray=200, n_iter=1, device=None)

    infinity_mask = np.isinf(new_points).any(-1)
    ori_points = ori_points[infinity_mask]# + ray_dir[infinity_mask] * 2e-4

    inverse_points = net.batch_ray_cast(ori_points, -ray_dir[infinity_mask], max_rays_per_chunk=10_000, ray_length=check_range * 1.8,
                       n_samples_along_ray=200, n_iter=1, device=None)

    new_points[infinity_mask] = inverse_points

    valid_mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
    valid_mask[v_mask] = np.isfinite(new_points).all(-1)

    mesh.vertices[valid_mask] = new_points[valid_mask[v_mask]]
    return mesh


def expand_faces(mesh, f_mask, num_iter=1):
    for i in range(num_iter):
        v_list = np.unique(mesh.faces[f_mask])
        v_mask = np.zeros(mesh.vertices.shape[0])
        v_mask[v_list] = True
        f_mask = v_mask[mesh.faces].any(-1)

    return f_mask


def remesh_by_view_direction(mesh, view_dir):
    # select faces that are close to be orthogonal to the DHF view
    remeshing_mask = np.abs((mesh.face_normals * view_dir).sum(-1)) < 0.3

    # swap remeshing faces to the back end
    faces_to_be_remeshed = mesh.faces[remeshing_mask]
    mesh.faces = mesh.faces[~remeshing_mask]
    fixed_face_count = mesh.faces.shape[0]
    mesh.faces = np.vstack([mesh.faces, faces_to_be_remeshed])

    mesh, new_v_mask = isotropic_explicit_remeshing(mesh, fixed_face_count)
    return mesh, new_v_mask
