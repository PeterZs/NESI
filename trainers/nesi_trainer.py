import os
import json

import numpy as np
import torch
import logging as log
import torch.nn.functional as F
import torch.optim as optim
from trainers.trainer import Trainer
from models import *
from datetime import datetime
from torch.utils.data import DataLoader
from utils.base_alg import compute_gradient
from datasets import *


class NESITrainer(Trainer):
    def __init__(self, args, args_str):
        super().__init__(args, args_str)

        self.set_path()
        self.set_lr_scheduler()

        with open(os.path.join(self.args.model_path, "args.json"), 'w') as outfile:
            json.dump(vars(self.args), outfile, indent=4)


    def set_path(self):
        # setup basic model path
        now = datetime.now()
        date_time = now.strftime("%m%d%H%M")
        suffix = date_time + "_" + str(self.args.net_width)

        if self.args.normal:
            suffix += "_n"
        if self.args.mask:
            suffix += "_m"

        self.args.model_path += "_" + suffix

        os.makedirs(self.args.model_path, exist_ok=True)

    def set_dataset(self):
        """
        Override this function if using a custom dataset.
        By default, it provides 2 datasets:

            AnalyticDataset
            MeshDataset

        The code uses the mesh dataset by default, unless --analytic is specified in CLI.
        """

        self.train_dataset = globals()[self.args.dataset](**vars(self.args))

        log.info("Dataset Size: {}".format(len(self.train_dataset)))

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=1,
                                            shuffle=True, pin_memory=True, num_workers=0)
        self.timer.check('create_dataloader')
        log.info("Loaded mesh dataset")


    def set_network(self):
        """
        Override this function if using a custom network, that does not use the default args based
        initialization, or if you need a custom network initialization scheme.
        """
        self.net = NESI(self.train_dataset.r_mat, z_origins=self.train_dataset.z_origins, **vars(self.args))
        if self.args.jit:
            self.net = torch.jit.script(self.net)

        if self.args.pretrained:
            self.net.load_state_dict(torch.load(self.args.pretrained))

        self.net.to(self.device)

        self.args.r_mat = self.train_dataset.r_mat.numpy().tolist()

        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))

    def set_renderer(self):
        """
        Override this function to use custom renderers.
        """
        # Renderer for logging
        self.log_tracer = None

    def set_optimizer(self):
        """
        Override this function to use custom optimizers. (Or, just add things to this switch)
        """
        from models.hf_net import HFNet
        param_list = list()
        for sub_net in self.net.mlp:
            if isinstance(sub_net, HFNet):
                param_list.append({
                    "params": sub_net.mlp_hf.parameters(),
                    "lr": self.args.lr,
                })
                param_list.append({
                    "params": sub_net.mlp_mask.parameters(),
                    "lr": 0.001,
                })
            else:
                param_list.append({
                    "params": sub_net.parameters(),
                    "lr": self.args.lr,
                })

        # Set geometry optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(param_list, lr=self.args.lr)
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(param_list, lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(param_list, lr=self.args.lr, momentum=0.8)
        else:
            raise ValueError('Invalid optimizer.')


    def set_lr_scheduler(self):
        if self.args.lr_scheduler == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_interval,
                                                        gamma=self.args.lr_decay)
        elif self.args.lr_scheduler == 'CosineAnnealing':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

    #######################
    # pre_epoch
    #######################

    def pre_epoch(self, epoch):
        """
        This function runs once before the epoch.
        """

        self.net.train()

        # Initialize the dict for logging
        self.log_dict['lr'] = 0
        self.log_dict['total_loss'] = 0
        self.log_dict['hf_loss'] = 0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['normal_loss'] = 0
        self.log_dict['dhf_penalty'] = 0
        self.log_dict['bce_loss'] = 0

        self.timer.check('pre_epoch done')


    #######################
    # iterate
    #######################b

    def iterate(self, epoch):
        """
        Override this if there is a need to override the dataset iteration.
        """

        self.net.zero_grad()
        loss = 0

        for n_iter, data in enumerate(self.train_data_loader):
            loss += self.step_geometry(epoch, n_iter, data)

        # Backpropagate
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    #######################
    # step
    #######################

    def step_geometry(self, epoch, n_iter, data, loss=0):
        """
        Override this function to change the per-iteration behaviour.
        """
        idx = n_iter + (epoch * self.dataset_size)
        log_iter = (idx % 100 == 0)

        # Map to device
        view_id = data[0]
        surface_uv = data[1].to(self.device)
        surface_h = data[2].to(self.device).squeeze()
        surface_normals = data[3].to(self.device).squeeze()
        spatial_uv = data[4].to(self.device).squeeze(0)

        if len(data) > 5:
            roi = data[5].to(self.device).squeeze()
            expanded_roi = data[6].to(self.device).squeeze()

        # Calculate loss
        surface_uv = surface_uv.requires_grad_(True)
        pred_h = self.net(view_id, surface_uv).squeeze()

        if view_id == 0:
            hf_loss = pred_h - surface_h
        else:
            pred_mask = pred_h[..., -1]
            pred_h = pred_h[..., 0]
            if self.args.mask:
                hf_loss = pred_h - surface_h

                hf_penalty = hf_loss.clone()
                hf_penalty[~expanded_roi] += 1e-2 # give this an offset
                hf_penalty = F.relu(hf_penalty[~roi]).mean()
                # protect when the tensor is empty
                if hf_penalty.isfinite():
                    loss += hf_penalty.mean()

                hf_loss = hf_loss[roi]
            else:
                hf_loss = pred_h - surface_h

        hf_loss = torch.abs(hf_loss).nanmean()
        # protect when the tensor is empty or all nan
        if hf_loss.isfinite():
            loss += hf_loss

        if self.args.normal:
            if surface_normals.shape[0] == 2:
                # Compute the gradient of HF
                dhf_gradients = compute_gradient(pred_h[..., 0], surface_uv).squeeze()

                # Compose normals
                normals = torch.concat([-dhf_gradients, torch.ones(dhf_gradients.shape[0]).to(self.device)[..., None]],
                                       dim=-1)
                valid_normal_mask = surface_normals[0].isfinite().all(-1)
                normal_loss = (1 - F.cosine_similarity(-F.normalize(normals[valid_normal_mask]),
                                                       surface_normals[0][valid_normal_mask], dim=-1)).mean()

                # Compute the gradient of HF
                dhf_gradients = compute_gradient(pred_h[..., 1], surface_uv).squeeze()

                # Compose normals
                normals = torch.concat([-dhf_gradients, torch.ones(dhf_gradients.shape[0]).to(self.device)[..., None]],
                                       dim=-1)
                valid_normal_mask = surface_normals[1].isfinite().all(-1)
                normal_loss += (1 - F.cosine_similarity(F.normalize(normals[valid_normal_mask]),
                                                        surface_normals[1][valid_normal_mask], dim=-1)).mean()
            else:
                hf_gradients = compute_gradient(pred_h, surface_uv).squeeze()
                # Compose normals
                normals = torch.concat([-hf_gradients, torch.ones(hf_gradients.shape[0]).to(self.device)[..., None]],
                                       dim=-1)
                if self.args.mask:
                    normal_loss = (1 - F.cosine_similarity(-F.normalize(normals), surface_normals, dim=-1)[roi]).mean()
                else:
                    normal_loss = (
                                1 - F.cosine_similarity(-F.normalize(normals), surface_normals, dim=-1)).mean()
            if normal_loss.isfinite():
                # normal_weight = 0.5 - np.cos(epoch * np.pi / float(self.args.epochs)) * 0.5
                normal_weight = np.tanh(epoch * 0.1 - float(self.args.epochs) * 0.04) * 0.5 + 0.5
                loss += normal_loss * 0.1 * normal_weight
            self.log_dict['normal_loss'] += normal_loss.item()


        bce = torch.nn.BCEWithLogitsLoss()  # with sigmoid integrated

        if view_id > 0:
            bce_loss = bce(pred_mask, torch.ones_like(pred_mask))

        pred_h = self.net(view_id, spatial_uv).squeeze()
        if view_id == 0:
            inf_penalty = F.relu(pred_h[..., 1] + 1e-2 - pred_h[..., 0])
            loss += inf_penalty.mean()
        else:
            bce_loss += bce(pred_h[..., 1], torch.zeros_like(pred_h[..., 1]))
            self.log_dict['bce_loss'] += bce_loss.item()
            loss += bce_loss

        #
        #     if self.args.alpha:
        #         alpha = (np.cos(epoch / float(self.args.epochs) * np.pi) + 1.0) / 4.0 + 0.5 # 1 -> 0.5
        #         # self.lr_scheduler.get_last_lr()[0] / self.args.lr
        #         if not self.args.training_only:
        #             self.log_dict['alpha'] = alpha
        #         loss += dhf_diff * alpha + normal_loss * self.args.normal_weight * (1 - alpha)
        #     else:
        #         loss += dhf_diff + normal_loss * self.args.normal_weight
        #
        #     if not self.args.training_only:
        #         self.log_dict['normal_loss'] += normal_loss.item()
        #
        # else:
        #     loss += dhf_diff

        # Update logs
        self.log_dict['total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += self.args.batch_size
        self.log_dict['hf_loss'] += hf_loss.item()
        self.log_dict['lr'] = self.optimizer.param_groups[-1]['lr']#self.lr_scheduler.get_last_lr()[0]

        # loss /= batch_size
        # # Backpropagate
        # loss.backward()
        # self.optimizer.step()

        return loss


    def log_tb(self, epoch):
        """
        Override this function to change loss logging.
        """
        # Average over iterations
        log_text = 'EPOCH {}/{}'.format(epoch+1, self.args.epochs)
        log_text += ' | lr: {:>.3E}'.format(self.log_dict['lr'])

        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])

        log_text += ' | HF loss: {:>.3E}'.format(self.log_dict['hf_loss'])

        if self.args.normal:
            log_text += ' | Normal loss: {:>.3E}'.format(self.log_dict['normal_loss'])
            self.writer.add_scalar('Loss/normal_loss', self.log_dict['normal_loss'], epoch)

        log_text += ' | BCE loss: {:>.3E}'.format(self.log_dict['bce_loss'])
        self.writer.add_scalar('Loss/bce_loss', self.log_dict['bce_loss'], epoch)

        # self.writer.add_scalar('Loss/hf_diff_loss', self.log_dict['hf_diff_loss'], epoch)
        # self.writer.add_scalar('blending_weights', self.log_dict['alpha'], epoch)

        self.writer.add_scalar('lr', self.log_dict['lr'], epoch)

        log.info(log_text)

        # Log losses
        self.writer.add_scalar('Loss/total_loss', self.log_dict['total_loss'], epoch)

    #######################
    # post_epoch
    #######################

    def post_epoch(self, epoch):
        """
        Override this function to change the post-epoch post processing.

        By default, this function logs to Tensorboard, renders images to Tensorboard, saves the model,
        and resamples the dataset.

        To keep default behaviour but also augment with other features, do

          super().post_epoch(self, epoch)

        in the derived method.
        """
        self.net.eval()

        self.log_tb(epoch)
        if epoch % self.args.save_every == 0:
            self.save_model(epoch)

        self.timer.check('post_epoch done')