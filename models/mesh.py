import torch
import point_cloud_utils as pcu
import numpy as np
import trimesh
import igl
#from pysdf import SDF
from utils.base_alg import normalize

class Mesh:
    def __init__(self, mesh):
        self.mesh = mesh
        self.intersector = pcu.RayMeshIntersector(self.mesh.vertices, self.mesh.faces)

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

    def batch_ray_cast(self, ray_origins, ray_directions, max_rays_per_chunk=10_000, ray_length=4.0,
                       n_samples_along_ray=100, n_iter=3, device=None):
        if isinstance(ray_origins, torch.Tensor):
            ray_origins = ray_origins.cpu().numpy()
        if isinstance(ray_directions, torch.Tensor):
            ray_directions = ray_directions.cpu().numpy()

        normalized_ray_directions = normalize(ray_directions)

        # fid is the index of each face intersected (-1 for ray miss)
        # bc are the barycentric coordinates of each intersected ray
        # t are the distances from the ray origin to the intersection for each ray (inf for ray miss)
        fid, bc, t = self.intersector.intersect_rays(ray_origins, normalized_ray_directions)

        hit_points = ray_origins + normalized_ray_directions * t[..., None]

        return hit_points

