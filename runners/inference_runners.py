import os
import json
import wandb
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm.auto import tqdm
from io import BytesIO
import torch.nn.functional as F

from PIL import Image
from pathlib import Path

from utils.class_registry import ClassRegistry
from datasets.datasets import ImageDataset
from datasets.transforms import transforms_registry
from utils.common_utils import tensor2im
from runners.base_runner import BaseRunner
from training.loggers import BaseTimer
from utils.common_utils import get_keys
from datasets.datasets import AttrDataset
from metrics.metrics import metrics_registry


inference_runner_registry = ClassRegistry()


@inference_runner_registry.add_to_registry(name="base_inference_runner")
class BaseInferenceRunner(BaseRunner):
    def run(self):
        self.run_inversion()
        self.run_editing()

    @torch.inference_mode()
    def run_inversion(self):
        output_inv_dir =  Path(self.config.exp.output_dir) / 'inversion'
        output_inv_dir.mkdir(parents=True, exist_ok=True)


        transform = transforms_registry[self.config.data.transform]().get_transforms()[
            "transform_inference"
        ]
        dataset = ImageDataset(self.config.data.inference_dir, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.model.workers,
        )

        self.method_results = []
        self.paths = dataset.paths
        self.method.eval()

        print("Start inversion")
        global_i = 0

        for input_batch in tqdm(dataloader):
            input_cuda = input_batch.to(self.device).float()

            images, result_batch = self._run_on_batch(input_cuda)
            result_batch["img_names"] = []
            
            for tensor in images:
                image = tensor2im(tensor)
                img_name = os.path.basename(dataset.paths[global_i])
                result_batch["img_names"].append(img_name)
                image.save(output_inv_dir / img_name)
                global_i += 1

            self.method_results.append(result_batch)


    @torch.inference_mode()
    def run_editing(self):
        editing_data = self.config.inference.editings_data

        for editing_name, editing_degrees in editing_data.items():
            print()
            print(editing_name, len(editing_degrees))
            output_edit_dir =  Path(self.config.exp.output_dir) / editing_name
            output_edit_paths = []

            for editing_degree in editing_degrees:
                editing_dir_degree_pth = output_edit_dir / f"edit_power_{editing_degree:.4f}"
                editing_dir_degree_pth.mkdir(parents=True, exist_ok=True)
                output_edit_paths.append(editing_dir_degree_pth)

            print(output_edit_paths)
            for method_res_batch in tqdm(self.method_results):
                print(method_res_batch['latents'].shape)
                edited_imgs_batch = self._run_editing_on_batch(
                    method_res_batch, editing_name, editing_degrees
                )
                print(edited_imgs_batch.shape)
                print(method_res_batch['img_names'])

                for edited_imgs, img_name in zip(edited_imgs_batch, method_res_batch['img_names']):
                    for edited_img_tensor, save_dir in zip(edited_imgs, output_edit_paths):
                        edited_img = tensor2im(edited_img_tensor)
                        edited_img.save(save_dir / img_name)


    def _run_on_batch(self, inputs):
        raise NotImplementedError()

    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees):
        raise NotImplementedError()


@inference_runner_registry.add_to_registry(name="psp_inference_runner")
class PSPInferenceRunner(BaseInferenceRunner):
    def _run_on_batch(self, inputs):
        images, latents = self.method(
            inputs,
            randomize_noise=False,
            resize=self.method_config.resize_outputs,
            return_latents=True
        )

        return images, {"w_latents": latents}

    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees=None):
        orig_latents = method_res_batch["w_latents"]
        edited_images = []

        for latent in orig_latents:
            edited_latents = self.get_edited_latent(
                latent.unsqueeze(0), editing_name, editing_degrees
            )
            if edited_latents is None:
                print(f'WARNING, skip editing {editing_name}')
                continue
            if type(edited_latents) == tuple:
                # edited_latents in stylespace case
                # edited_latents: (style_main, style_rgb)
                # style_main: [torch.tensor(len(editing_degrees) x latent_len), ...], len = 18
                # style_rgb: [torch.tensor(len(editing_degrees) x latent_len), ...], len = 7
                image_edits = self.method.decoder(edited_latents, is_stylespace=True, input_is_latent=True)[0]
                edited_images.append(image_edits)
            else:
                # edited_latents in w or w+ space case
                # edited_latents : [torch.tensor(1 x 18 x 512), ...], len = len(edited_latents)
                edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)
                image_edits = self.method.decoder(edited_latents, input_is_latent=True)[0]
                edited_images.append(image_edits)
        edited_images = torch.stack(edited_images)

        return edited_images  # : torch.tensor(batch_size x len(powers) x pics)



@inference_runner_registry.add_to_registry(name="fse_inference_runner")
class FSEInferenceRunner(BaseInferenceRunner):
    def _run_on_batch(self, inputs):
        images, w_recon, fused_feat, predicted_feat = self.method(inputs, return_latents=True)
        
        x = F.interpolate(inputs, size=(256, 256), mode="bilinear", align_corners=False)
        w_e4e = self.method.e4e_encoder(x)
        w_e4e = w_e4e + self.method.latent_avg
        
        return images, {"latents": w_recon, 
                        "fused_feat": fused_feat, 
                        "predicted_feat": predicted_feat, 
                        "w_e4e": w_e4e}
          
    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees):
        orig_latents = method_res_batch["latents"]
        edited_images = []
        n_iter = 1e5

        for i, latent in enumerate(orig_latents):
            edited_latents = self.get_edited_latent(
                latent.unsqueeze(0), editing_name, editing_degrees
            )
            
            w_e4e = method_res_batch["w_e4e"][i].unsqueeze(0)
            edited_w_e4e = self.get_edited_latent(
                w_e4e, editing_name, editing_degrees
            )

            if edited_latents is None or edited_w_e4e is None:
                print(f'WARNING, skip editing {editing_name}')
                continue

            is_stylespace = isinstance(edited_latents, tuple)
            if not is_stylespace:
                edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)
                edited_w_e4e = torch.cat(edited_w_e4e, dim=0).unsqueeze(0)

            w_e4e = w_e4e.repeat(len(editing_degrees), 1, 1)  # bs = len(editing_degrees)
            w_latent = latent.unsqueeze(0).repeat(len(editing_degrees), 1, 1)

            _, fs_x = self.method.decoder(
                [w_e4e],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True,
                early_stop=64
            )

            _, fs_y = self.method.decoder(
                edited_w_e4e,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                is_stylespace=is_stylespace,
                return_features=True,
                early_stop=64
            )

            delta = fs_x[9] - fs_y[9]

            fused_feat = method_res_batch["fused_feat"][i].to(self.device)
            fused_feat = fused_feat.repeat(len(editing_degrees), 1, 1, 1)


            edited_feat = self.method.encoder(torch.cat([fused_feat, delta], dim=1))  # encoder == feature editor
            edit_features = [None] * 9 + [edited_feat] + [None] * (17 - 9)

            image_edits, _ = self.method.decoder(edited_latents,
                                                   input_is_latent=True,
                                                   features_in=edit_features,
                                                   feature_scale=min(1.0, 0.0001 * n_iter),
                                                   is_stylespace=is_stylespace,
                                                   randomize_noise=False)

            edited_images.append(image_edits)
        edited_images = torch.stack(edited_images)

        return edited_images  # : torch.tensor(batch_size x len(powers) x pics)


@inference_runner_registry.add_to_registry(name="fse_e4e_inference_runner")
class FSEE4EInferenceRunner(BaseInferenceRunner):
    def _run_on_batch(self, inputs):
        images, w_recon, fused_feat, predicted_feat = self.method(inputs, return_latents=True)
        
        x = F.interpolate(inputs, size=(256, 256), mode="bilinear", align_corners=False)
        w_e4e = self.method.e4e_encoder(x)
        w_e4e = w_e4e + self.method.latent_avg
        
        return images, {"latents": w_recon, 
                        "fused_feat": fused_feat, 
                        "predicted_feat": predicted_feat, 
                        "w_e4e": w_e4e}
          
    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees):
        orig_latents = method_res_batch["latents"]
        edited_images = []
        n_iter = 1e5
        print("W_e4e", method_res_batch["w_e4e"].shape)

        for i, latent in enumerate(orig_latents):
            edited_latents = self.get_edited_latent(
                latent.unsqueeze(0), editing_name, editing_degrees
            )
            
            w_e4e = method_res_batch["w_e4e"][i].unsqueeze(0)
            edited_w_e4e = self.get_edited_latent(
                w_e4e, editing_name, editing_degrees
            )

            if edited_latents is None or edited_w_e4e is None:
                print(f'WARNING, skip editing {editing_name}')
                continue

            is_stylespace = isinstance(edited_latents, tuple)
            if not is_stylespace:
                edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)
                edited_w_e4e = torch.cat(edited_w_e4e, dim=0).unsqueeze(0)

            if len(edited_latents) != 2:
                print("edited_latents", edited_latents.shape)
                print(edited_w_e4e.shape)

            w_e4e = w_e4e.repeat(len(editing_degrees), 1, 1)  # bs = len(editing_degrees)
            w_latent = latent.unsqueeze(0).repeat(len(editing_degrees), 1, 1)

            _, fs_x = self.method.decoder(
                [w_e4e],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True,
                early_stop=64
            )

            image_edits, fs_y = self.method.decoder(
                edited_w_e4e,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                is_stylespace=is_stylespace,
                return_features=True,
            )

            edited_images.append(image_edits)
        edited_images = torch.stack(edited_images)

        return edited_images  # : torch.tensor(batch_size x len(powers) x pics)


@inference_runner_registry.add_to_registry(name="fse_inverter_inference_runner")
class FSEInverterInferenceRunner(BaseInferenceRunner):
    def _run_on_batch(self, inputs):
        images, w_recon, fused_feat, predicted_feat = self.method(inputs, return_latents=True)
        
        return images, {"latents": w_recon, 
                        "fused_feat": fused_feat, 
                        "predicted_feat": predicted_feat}
          
    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees):
        orig_latents = method_res_batch["latents"]
        edited_images = []
        n_iter = 1e5

        for i, latent in enumerate(orig_latents):
            edited_latents = self.get_edited_latent(
                latent.unsqueeze(0), editing_name, editing_degrees
            )

            if edited_latents is None or edited_w_e4e is None:
                print(f'WARNING, skip editing {editing_name}')
                continue

            is_stylespace = isinstance(edited_latents, tuple)
            if not is_stylespace:
                edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)

            w_latent = latent.unsqueeze(0).repeat(len(editing_degrees), 1, 1)

            fused_feat = method_res_batch["fused_feat"][i].to(self.device)
            fused_feat = fused_feat.repeat(len(editing_degrees), 1, 1, 1)

            edit_features = [None] * 9 + [fused_feat] + [None] * (17 - 9)

            image_edits, _ = self.method.decoder(edited_latents,
                                               input_is_latent=True,
                                               features_in=edit_features,
                                               feature_scale=min(1.0, 0.0001 * n_iter),
                                               is_stylespace=is_stylespace,
                                               randomize_noise=False)

            edited_images.append(image_edits)
        edited_images = torch.stack(edited_images)

        return edited_images  # : torch.tensor(batch_size x len(powers) x pics)

