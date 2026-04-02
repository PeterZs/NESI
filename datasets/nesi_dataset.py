import os

import torch
from torch.utils.data import Dataset

import numpy as np

from utils.timer import PerfTimer#, setparam

class NESIDataset(Dataset):
    """Base class for NESI datasets."""

    def __init__(self,
        dataset_path = None,
        batch_size = 512,
        **kwargs
    ):
        #self.args = args
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.n_surface_samples = batch_size // 2
        self.n_spatial_samples = self.batch_size - self.n_surface_samples

        self.r_mat = [] # points @ r_mat -> nesi local coordinates
        self.z_origins = []
        self.surface_uv = []
        self.surface_h = []
        self.surface_normals = []
        self.spatial_uv = []
        self.roi = []
        self.expanded_roi = []

        for path, dirc, files in os.walk(self.dataset_path):
            for name in sorted(files):
                if name.endswith('.npz'):

                    npz = np.load(os.path.join(path, name))
                    self.r_mat += [torch.Tensor(npz["r_mat"])]

                    self.surface_uv += [torch.Tensor(npz["surface_samples"][..., :2]).float()]
                    self.surface_h += [torch.Tensor(npz["surface_samples"][..., 2:]).float()]

                    self.surface_normals += [torch.Tensor(npz["surface_normals"]).float()]
                    if self.surface_normals[-1].dim() == 2:
                        self.surface_normals[-1] = self.surface_normals[-1][None, ...]

                    self.spatial_uv += [torch.Tensor(npz["spatial_samples"][..., :2]).float()]

                    try:
                        self.roi += [torch.Tensor(npz["roi_mask"]).bool()]
                        self.expanded_roi += [torch.Tensor(npz["expanded_roi_mask"]).bool()]
                    except:
                        self.roi += [None]
                        self.expanded_roi += [None]

                    try:
                        self.z_origins += [float(npz["z_origin"])]
                    except:
                        self.z_origins += [-2]

        self.r_mat = torch.stack(self.r_mat)


    def __getitem__(self, view_id: int):
        random_idx = torch.randint(0, self.surface_uv[view_id].shape[0], (self.n_surface_samples,))#(torch.rand(surface_sampling) * self.surface_uv[idx].shape[0]).long()
        surface_uv = torch.index_select(self.surface_uv[view_id], 0, random_idx)
        surface_h = torch.index_select(self.surface_h[view_id], 0, random_idx)
        surface_normals = torch.index_select(self.surface_normals[view_id], 1, random_idx)

        sp_random_idx = torch.randint(0, self.spatial_uv[view_id].shape[0], (self.n_spatial_samples,))#(torch.rand(surface_sampling) * self.surface_uv[idx].shape[0]).long()
        spatial_uv = torch.index_select(self.spatial_uv[view_id], 0, sp_random_idx)

        if view_id == 0:
            return view_id, surface_uv, surface_h, surface_normals, spatial_uv
        else:
            roi = torch.index_select(self.roi[view_id], 0, random_idx)
            expanded_roi = torch.index_select(self.expanded_roi[view_id], 0, random_idx)
            return view_id, surface_uv, surface_h, surface_normals, spatial_uv, roi, expanded_roi


            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        return self.r_mat.shape[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1
