import torch
import point_cloud_utils as pcu
import numpy as np
import trimesh
from utils.base_alg import align_vectors, normalize, point_mesh_distances
import torch.nn as nn
from pykdtree.kdtree import KDTree


class ESI(nn.Module):
    def __init__(
            self,
            view_dirs, mesh, dhf_ids=[], z_origins=None, device=torch.device("cpu")
    ):
        """
        Creates a new ESI instance, which maps 2D input coordinates + view ID to depth values.

        This explicit-shape-intersection consists of:
            * View directions
            * A z offset in camera coordinates
            * The target mesh

        This explicit-shape-intersection supports:
            * Shape approximation with multi-view depth map
            * Inside-outside query for ESI

        Args:
            view_dirs: The view direction array, should be in shape (N, 3), where N is the number of views;
            mesh: The target mesh;
            dhf_ids: A list of DHF IDs of view direction;
            z_offset: The z offset in camera coordinates;
            device: The device on which to run the computations;
        """
        super(ESI, self).__init__()

        if isinstance(view_dirs, torch.Tensor):
            self.view_dirs = view_dirs.detach().cpu().numpy()
        else:
            self.view_dirs = view_dirs

        if z_origins is None:
            self.z_origins = np.array([-2.0] * view_dirs.shape[0])
        else:
            self.z_origins = z_origins

        z_axis = np.array([0., 0., 1.], dtype=np.float64)

        # Compute rotation matrix s.t. view_dir @ R = [0, 0, 1]
        self.r_mat = [align_vectors(view_dir, z_axis).T for view_dir in self.view_dirs]
        self.r_mat = np.stack(self.r_mat)

        self.device = device

        self.mesh = mesh
        self.intersector = pcu.RayMeshIntersector(self.mesh.vertices, self.mesh.faces)
        self.convex_hull = trimesh.convex.convex_hull(self.mesh)
        self.ch_intersector = pcu.RayMeshIntersector(self.convex_hull.vertices, self.convex_hull.faces)
        self.dhf_ids = dhf_ids

    def forward(self, view_id, xy):
        """
        For given view id and xy query points, the function will return the height of those points
        :param view_id: View ID
        :param xy: 2D input coordinates, whose shape is (N, 2)
        :return: Height of the input points, whose shape is (N, 2) for DHF views, or (N, 1) for HF views.
        """
        if isinstance(xy, torch.Tensor):
            xy = xy.detach().cpu().numpy()

        # Convert the 2D input [x, y] to [x, y, -z_offset] in local coordinates (we suppose the shape is in [-1, 1]^3 in world coordinates)
        ray_o_local = np.hstack([xy, np.full((xy.shape[0], 1), self.z_origins[view_id])])

        # Convert NDC to the world coordinate
        ray_o_world = ray_o_local @ self.r_mat[view_id].T

        view_dir = np.tile(self.view_dirs[view_id][np.newaxis, :], (ray_o_world.shape[0], 1))

        # fid is the index of each face intersected (-1 for ray miss)
        # bc are the barycentric coordinates of each intersected ray
        # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
        fid, bc, t = self.intersector.intersect_rays(ray_o_world, view_dir)

        # abs() version may problematic!
        t += self.z_origins[view_id]

        # when only one point, the returned variables should be reshaped
        if fid.size == 1:
            fid = fid.reshape(1)
            t = t.reshape(1)
        t = t[..., np.newaxis]

        if view_id in self.dhf_ids:
            # DHF: move the origin points forward the view direction and look back
            _, _, t_far = self.intersector.intersect_rays(ray_o_world + view_dir * abs(self.z_origins[view_id]) * 2, -view_dir)

            # revert the distance and remove the offset
            t_far = -t_far + abs(self.z_origins[view_id])

            # stack the height values together in the order [near, far]
            t = np.stack([t.squeeze(-1), t_far], axis=-1)

        # t is [N, 1] (HF), or [N, 2] (DHF)
        return t


    def is_visible(self, points):
        """
        Check the ESI visibility of a set of points.
        The input points should be off the surface (or shifted). Otherwise, the function will return random results.
        :param points:
        :return:
        """
        from utils.base_alg import is_visible
        # check visibility
        visible_mask = [is_visible(self.view_dirs[i], points, self.intersector, is_dhf=i in self.dhf_ids) for i in range(self.view_dirs.shape[0])]
        visible_mask = np.stack(visible_mask, axis=0)
        visible_mask = np.any(visible_mask, axis=0)
        return visible_mask


    def get_visible_score(self, points, normals):
        from utils.base_alg import is_visible
        shifted_points = points + normals * 1e-5
        # check visibility
        visible_mask = [is_visible(self.view_dirs[i], shifted_points, self.intersector, is_dhf=i in self.dhf_ids) for i in range(self.view_dirs.shape[0])]
        visible_mask = np.stack(visible_mask, axis=0)

        # check if the points are well visible from previous ESI
        visible_score = (normals[None, ...] * -self.view_dirs[:, None, :]).sum(-1)
        # set the points both side visible for DHF views
        visible_score[0] = np.abs(visible_score[0])
        visible_score[~visible_mask] = -1
        visible_score = np.max(visible_score, axis=0)

        return visible_score


    def mask_from_views(self, points, normals):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(normals, torch.Tensor):
            normals = normals.cpu().numpy()

        #mask_list = []
        normal_list = []

        shifted_points = (points + normals * 1e-4).astype(np.float64)

        for i in range(self.view_dirs.shape[0]):
            # Compute visibility for the HF view:
            if (i in self.dhf_ids):
                esi = ESI(self.view_dirs[i][None, ...], self.mesh, dhf_ids=[i])
                front_normal_score = (normals * -self.view_dirs[i]).sum(-1)
                back_normal_score = (normals * self.view_dirs[i]).sum(-1)
                normal_score = np.where(front_normal_score > back_normal_score, front_normal_score, back_normal_score)
            else:
                esi = ESI(self.view_dirs[i][None, ...], self.mesh, dhf_ids=[])
                normal_score = (normals * -self.view_dirs[i]).sum(-1)
            #mask_list += [dhf_esi.is_visible(shifted_points)]
            normal_score[~esi.is_visible(shifted_points)] = -1
            normal_list += [normal_score]

        normal_list = np.stack(normal_list, axis=0)

        view_mask = np.zeros_like(normal_list, dtype=bool)

        view_mask[np.argmax(normal_list, axis=0), np.arange(view_mask.shape[1])] = 1
        view_mask[:, normal_list.max(0) <= -0.5] = 0

        return view_mask


    def mask_by_nesi_views(self, points, normals):

        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(normals, torch.Tensor):
            normals = normals.cpu().numpy()

        mask_list = []

        shifted_points = (points + normals * 1e-5).astype(np.float64)
        # Compute visibility for the DHF view:
        dhf_esi = ESI(self.view_dirs[0][None, ...], self.mesh, dhf_ids=[0])
        mask_list += [dhf_esi.is_visible(shifted_points)]

        for i in range(1, self.view_dirs.shape[0]):
            # Compose ESI using previous view directions
            prev_esi = ESI(self.view_dirs[:i], self.mesh, dhf_ids=[0])

            # check if the points are visible from the previous ESI
            visible_mask = prev_esi.is_visible(shifted_points)

            # Set ray origins, only check those visible points
            ray_origin_lc = (points @ self.r_mat[i])[visible_mask]
            ray_origin_lc[..., 2] = self.z_origins[i]
            ray_origin_wc = ray_origin_lc @ self.r_mat[i].T

            # Set the view direction
            view_dirs = ray_origin_wc - shifted_points[visible_mask]
            esi_intersection_points = prev_esi.batch_ray_cast(shifted_points[visible_mask], view_dirs, ray_length=1.0)

            # If the ray hits infinity, then the whole ray is visible (not intersect to the previous ESI); otherwise it is not visible
            visible_mask[visible_mask.copy()] = np.isinf(esi_intersection_points).any(-1)  # dis < 1e-4

            # check if the points are well visible from previous ESI
            view_score = prev_esi.get_visible_score(points[visible_mask], normals[visible_mask])
            # a point is well visible if it is not in the grazing region of ESI
            well_visible_mask = view_score > np.cos(70.0 * np.pi / 180.0)

            # and if current view score is not better than the previous ones in grazing region
            cur_view_score = ((normals @ self.r_mat[i])[visible_mask] * np.array([0, 0, -1])).sum(-1)
            well_visible_mask = np.bitwise_or(well_visible_mask, np.bitwise_and(view_score <= np.cos(70.0 * np.pi / 180.0),
                                                                                cur_view_score < view_score))

            visible_mask[visible_mask.copy()] = well_visible_mask
            roi_mask = ~visible_mask

            # Check if those points are visible from current ESI view direction
            cur_esi =  ESI(self.view_dirs[i][None, ...], self.mesh, dhf_ids=[])
            roi_mask = roi_mask & cur_esi.is_visible(shifted_points)

            # Dilate the supervised region
            if np.count_nonzero(roi_mask) > 0:
                gt_tree = KDTree(points[roi_mask])
                dis, _ = gt_tree.query(points, k=1)
                roi_mask = (dis < 5e-3).squeeze()
            else:
                print("no valid point from this view direction")

            mask_list += [roi_mask]
        return mask_list



    def ray_cast(self, ray_origins, ray_directions, ray_length=4.0, n_iter=3, n_samples_along_ray=100):
        # initialize the first hits
        first_hits = np.full(ray_origins.shape, np.inf)

        # refining steps
        for j in range(n_iter):
            # evenly sample points on the ray
            t = np.linspace(0, ray_length, n_samples_along_ray + 1)
            t = np.tile(t[None], (ray_origins.shape[0], 1))
            sample_points = ray_origins[:, None, :] + ray_directions[:, None, :] * t[:, :, None]

            if j == 0:
                # filter out invalid points to save inference time
                bbox_mask = np.all(sample_points >= -1, axis=-1) & np.all(sample_points <= 1, axis=-1)

                # Forward the network
                inside_mask = self.occupancy_check(sample_points[bbox_mask])

                is_inside = np.zeros(sample_points.shape[:-1], dtype=bool)
                is_inside[bbox_mask] = inside_mask

            else:
                # Forward the network
                is_inside = self.occupancy_check(sample_points.reshape(-1, 3))
                is_inside = is_inside.reshape(sample_points.shape[:-1])

            # The intersection index i means the intersection point lies in [i, i+1]
            has_intersection = is_inside[..., :-1] != is_inside[..., 1:]

            cur_hit_mask = np.any(has_intersection, axis=-1)
            # # The mask if the ray has intersection
            if j == 0:
                hit_mask = cur_hit_mask
            else:
                hit_mask[hit_mask.copy()] = cur_hit_mask
                #
            sample_points = sample_points[cur_hit_mask, ...]
            ray_directions = ray_directions[cur_hit_mask, ...]
            has_intersection = has_intersection[cur_hit_mask, ...]

            # Get the first intersection ID on each ray
            first_intersect_id = np.argmax(has_intersection.astype(np.float32), axis=1)

            # Shift the origin points of rays
            ray_origins = sample_points[np.arange(sample_points.shape[0]), first_intersect_id]

            ray_length = ray_length / n_samples_along_ray

        first_hits[hit_mask] = ray_origins + ray_directions * ray_length * 0.5
        return first_hits


    def batch_ray_cast(self, ray_origins, ray_directions, max_rays_per_chunk=10_000, ray_length=4.0,
                       n_samples_along_ray=100, n_iter=3, use_convex_hull=False, device=None):

        if isinstance(ray_origins, torch.Tensor):
            ray_origins = ray_origins.cpu().numpy()
        if isinstance(ray_directions, torch.Tensor):
            ray_directions = ray_directions.cpu().numpy()

        esi_intersections = np.zeros_like(ray_origins)

        if use_convex_hull:
            # check if the ray shoots to infinity by using the convex hull
            normalized_ray_directions = normalize(ray_directions)
            _, _, dis_to_o = self.ch_intersector.intersect_rays(ray_origins, normalized_ray_directions)
            ch_finite_mask = np.isfinite(dis_to_o)

            ray_origins = ray_origins[ch_finite_mask]
            ray_directions = ray_directions[ch_finite_mask]
            esi_intersections[~ch_finite_mask] = np.inf

        # split the array into sub-arrays to save memory
        chunks = ray_origins.shape[0] // max_rays_per_chunk + 1
        ray_o_chunks = np.array_split(ray_origins, chunks)
        ray_d_chunks = np.array_split(ray_directions, chunks)
        hits = []
        from tqdm import tqdm
        pbar = tqdm(total=chunks)
        for ray_o, ray_d in zip(ray_o_chunks, ray_d_chunks):
            hits += [self.ray_cast(ray_o, ray_d, ray_length, n_iter, n_samples_along_ray)]
            pbar.update(1)
        pbar.close()
        hits = np.concatenate(hits, axis=0)

        if use_convex_hull:
            # Set the hit point as the center point of the segment
            # ch_finite_mask[ch_finite_mask.copy()] = esi_checking_mask
            esi_intersections[ch_finite_mask] = hits
        else:
            esi_intersections = hits

        return esi_intersections


    def get_n_views(self):
        return self.r_mat.shape[0]

    def get_normals(self, pts):
        # Ideally, pts should be exactly on the surface
        # Find the closest point on the mesh to each random point
        (closest_points,
         distances,
         triangle_id) = self.gt_mesh.nearest.on_surface(pts.cpu().numpy())

        # Compute barycentric coordinates in the triangle
        bary = trimesh.triangles.points_to_barycentric(triangles=self.gt_mesh.triangles[triangle_id], points=closest_points)
        # Compute normals of points
        p_normal = trimesh.unitize((self.gt_mesh.vertex_normals[self.gt_mesh.faces[triangle_id]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
        p_normal = torch.Tensor(p_normal).to(pts.device)
        return p_normal


    def occupancy_check(self, xyz):

        if isinstance(xyz, torch.Tensor):
            device = xyz.device
            xyz = xyz.detach().cpu().numpy()

        # get the h-values from each view and append them to a list
        h_input = []
        h_esi = []
        for i in range(self.get_n_views()):
            #  convert to local coordinate
            xyz_local = xyz @ self.r_mat[i]
            h_input += [xyz_local[..., -1]]
            h_esi += [self.forward(i, xyz_local[..., :2])]

        # check if the depths from all views are smaller than the corresponding depths of the point, which indicates the point is inside
        is_inside = []
        for i in range(self.get_n_views()):
            if i in self.dhf_ids:
                is_inside += [
                    (h_input[i] > h_esi[i][..., 0]) & (h_input[i] < h_esi[i][..., 1]) & np.isfinite(h_esi[i]).all(-1)]
            else:
                is_inside += [((h_input[i] > h_esi[i][..., 0]) & np.isfinite(h_esi[i][..., 0])) | (h_input[i] < self.z_origins[i]) | (h_esi[i][..., 0] < self.z_origins[i])]

        is_inside = np.stack(is_inside)
        is_inside = np.all(is_inside, axis=0)

        if 'device' in locals():
            is_inside = torch.from_numpy(is_inside).to(device)

        return is_inside
