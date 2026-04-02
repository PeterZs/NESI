import numpy as np
import torch
import torch.nn as nn
from models.mlp import MLP
from models.hf_net import HFNet
from scipy.spatial.transform import Rotation as R
from utils.base_alg import get_best_projections

class NESI(nn.Module):
    def __init__(
            self,
            r_mat,
            z_origins=None,
            arch='MLP',
            net_width=30,
            output_dim=2,
            all_dhf=False,
            **kwargs
    ):#
        super(NESI, self).__init__()
        n_views = len(r_mat)

        self.mlp = nn.ModuleList()
        for i in range(n_views):
            if i == 0:
                self.mlp.append(MLP(net_width=int(net_width * 2.0), output_dim=output_dim, **kwargs))
            else:
                self.mlp.append(HFNet(net_width=net_width, **kwargs))

        self.register_buffer('r_mat', torch.Tensor(r_mat))

        self.r_vec = nn.Parameter(torch.Tensor(R.from_matrix(self.r_mat).as_rotvec()), requires_grad=False)
        if z_origins is None:
            z_origins = [-2.] * n_views
        self.z_origins = nn.Parameter(torch.Tensor(z_origins), requires_grad=False)

        self.dhf_id = 0


    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict.update({'r_mat': torch.Tensor(R.from_rotvec(state_dict['r_vec'].cpu()).as_matrix())})
        if 'z_cuts' in state_dict:
            state_dict.update({'z_origins': state_dict['z_cuts']})
            del state_dict['z_cuts']
        super().load_state_dict(state_dict, *args, **kwargs)
        return


    def forward(self, idx, input):
        return self.mlp[idx](input)


    def get_n_views(self):
        return len(self.r_mat)


    def occupancy_check(self, input, debug=False):
        # shape [views, num_pts, 3]
        coord_ndc = torch.bmm(input[None, ...].expand(self.r_mat.shape[0], -1, -1), self.r_mat)

        h_values = []
        for i in range(self.r_mat.shape[0]):
            h_values += [self.forward(i, coord_ndc[i, ..., :2]).squeeze(0).detach()]

        # check if the depth from all views smaller than the depth of the point, which indicates the point is inside
        inside_mask = []
        for i in range(self.r_mat.shape[0]):
            if i == self.dhf_id:
                inside_mask += [(coord_ndc[i, ..., -1] > h_values[i][..., 0])]
                inside_mask += [(coord_ndc[i, ..., -1] < h_values[i][..., 1])]
            else:
                object_mask = nn.Sigmoid()(h_values[i][..., -1].detach()) > 0.5
                h_values[i][~object_mask, 0] = 3.9

                # negative_mask = coord_ndc[i, ..., -1] <= 0
                # h_values[i][negative_mask, 0] = (coord_ndc[i, ..., -1] - 0.1)[negative_mask]
                inside_mask += [(coord_ndc[i, ..., -1] > h_values[i][..., 0]) |
                                (h_values[i][..., 0] < self.z_origins[..., i]) | (coord_ndc[i, ..., -1] < self.z_origins[..., i])]

        # if the point is viewed as outside from any view, it should be outside
        inside_mask = torch.stack(inside_mask)
        overall_inside_mask = torch.all(inside_mask, dim=0)

        occupancy = torch.zeros_like(overall_inside_mask).bool()
        occupancy[overall_inside_mask] = 1

        if debug:
            normals = self.get_normals(input)
            normals = torch.stack(normals).detach()
            return occupancy, inside_mask, normals

        return occupancy


    def get_normals(self, pts):
        from utils.base_alg import compute_gradient, normalize

        with torch.enable_grad():
            pts = pts.requires_grad_(True)

            # shape [views, num_pts, 3]
            coord_ndc = torch.bmm(pts[None, ...].expand(self.r_mat.shape[0], -1, -1), self.r_mat)

            net_inputs = []
            h_values = []
            for i in range(self.r_mat.shape[0]):
                net_inputs += [coord_ndc[i, ..., :2]]
                h_values += [self.forward(i, net_inputs[-1]).squeeze(0)]

            normals = []
            for i in range(self.r_mat.shape[0]):
                if i == self.dhf_id:
                    gradients = compute_gradient(h_values[0][..., 0], net_inputs[i]).detach()
                    normal = torch.concat([gradients, -torch.ones(gradients.shape[0]).to(gradients.device)[..., None]],
                                      dim=-1)
                    normal = normalize(normal)
                    normal = normal @ self.r_mat[i].T
                    normals += [normal]

                    gradients = compute_gradient(h_values[0][..., 1], net_inputs[i]).detach()
                    normal = torch.concat([-gradients, torch.ones(gradients.shape[0]).to(gradients.device)[..., None]],
                                      dim=-1)
                    normal = normalize(normal)
                    normal = normal @ self.r_mat[i].T
                    normals += [normal]
                else:
                    gradients = compute_gradient(h_values[i][..., 0], net_inputs[i]).detach()
                    normal = torch.concat([gradients, -torch.ones(gradients.shape[0]).to(gradients.device)[..., None]],
                                      dim=-1)
                    normal = normalize(normal)
                    normal = normal @ self.r_mat[i].T
                    normals += [normal]

        return normals


    def ray_cast(self, ray_origins, ray_directions, ray_length=4.0, n_iter=3, n_samples_along_ray=100):
        # initialize the first hits
        first_hits = torch.full(ray_origins.shape, torch.inf, device=ray_origins.device)

        # refining steps
        for j in range(n_iter):
            # evenly sample points on the ray
            t = torch.linspace(0, ray_length, n_samples_along_ray + 1, device=ray_origins.device)
            t = t[None].expand(ray_origins.shape[0], -1)
            sample_points = ray_origins[:, None, :] + ray_directions[:, None, :] * t[:, :, None]

            if j == 0:
                # filter out invalid points to save inference time
                bbox_mask = (sample_points >= -1).all(dim=-1) & (sample_points <= 1).all(dim=-1)

                # Forward the network
                inside_mask = self.occupancy_check(sample_points[bbox_mask])

                is_inside = torch.zeros(sample_points.shape[:-1], dtype=torch.bool, device=ray_origins.device)
                is_inside[bbox_mask] = inside_mask

            else:
                # Forward the network
                is_inside = self.occupancy_check(sample_points.reshape(-1, 3))
                is_inside = is_inside.reshape(sample_points.shape[:-1])

            # The intersection index i means the intersection point lies in [i, i+1]
            has_intersection = is_inside[..., :-1] != is_inside[..., 1:]

            # The mask if the ray has intersection
            cur_hit_mask = has_intersection.any(dim=-1)
            if j == 0:
                hit_mask = cur_hit_mask
            else:
                hit_mask[hit_mask.clone()] = cur_hit_mask

            sample_points = sample_points[cur_hit_mask, ...]
            ray_directions = ray_directions[cur_hit_mask, ...]
            has_intersection = has_intersection[cur_hit_mask, ...]

            # Get the first intersection ID on each ray
            first_intersect_id = torch.argmax(has_intersection.float(), dim=1)

            # Shift the origin points of rays
            ray_origins = sample_points[torch.arange(sample_points.shape[0]), first_intersect_id]

            ray_length = ray_length / n_samples_along_ray

        first_hits[hit_mask] = ray_origins + ray_directions * ray_length * 0.5
        return first_hits


    def batch_ray_cast(self, ray_origin, ray_direction, max_rays_per_chunk=10_000, ray_length=4.0,
                       n_samples_along_ray=100, n_iter=2, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        chunks = ray_origin.shape[0] // max_rays_per_chunk + 1

        ray_o_chunks = torch.chunk(torch.Tensor(ray_origin), chunks=chunks)
        ray_d_chunks = torch.chunk(torch.Tensor(ray_direction), chunks=chunks)

        intersect_points = []
        from tqdm import tqdm
        pbar = tqdm(total=chunks)
        for ray_o, ray_d in zip(ray_o_chunks, ray_d_chunks):
            intersect_points += [
                self.ray_cast(ray_o.to(device), ray_d.to(device), ray_length=ray_length,
                          n_samples_along_ray=n_samples_along_ray, n_iter=n_iter)]
            pbar.update(1)
        pbar.close()
        intersect_points = torch.concat(intersect_points, dim=0).detach().cpu().numpy()
        return intersect_points


