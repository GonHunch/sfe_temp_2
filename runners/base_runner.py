import os
import sys
import torch
import json
import omegaconf
import wandb
import glob

from pathlib import Path
from editings.latent_editor_wrapper import LatentEditorWrapper

from models.methods import methods_registry
from metrics.metrics import metrics_registry
from utils.model_utils import get_stylespace_from_w


class BaseRunner:
    def __init__(self, config):
        self.config = config
        self.method_config = config.methods_args[config.model.method]

    def setup(self):
        self._setup_device()
        self._setup_latent_editor()
        self._setup_method()

    def get_edited_latent(self, original_latent, editing_name, editing_degrees):

        if editing_name in self.latent_editor.stylespace_directions:
            stylespace_latent = get_stylespace_from_w(original_latent, self.method.decoder)
            edited_latents = (
                self.latent_editor.get_stylespace_edits_with_direction(
                    stylespace_latent, editing_degrees, editing_name
                ))
        elif editing_name in self.latent_editor.interfacegan_directions:
            edited_latents = (
                self.latent_editor.get_single_interface_gan_edits_with_direction(
                    original_latent, editing_degrees, editing_name
                ))

        elif editing_name in self.latent_editor.styleclip_meta_data:
            edited_latents = self.latent_editor.get_single_styleclip_latent_mapper_edits_with_direction(
                original_latent, editing_degrees, editing_name
            )

        elif editing_name in self.latent_editor.ganspace_directions:
            edited_latents = (
                self.latent_editor.get_single_ganspace_edits_with_direction(
                    original_latent, editing_degrees, editing_name
                )
            )
        elif editing_name in self.latent_editor.fs_directions.keys():
            edited_latents = self.latent_editor.get_fs_edits_with_direction(
                    original_latent, editing_degrees, editing_name
                )
        elif editing_name in ["bangs_sc"]:
            stylespace_latent = get_stylespace_from_w(original_latent, self.method.decoder)
            edited_latents = (
                self.latent_editor.get_single_styleclip_latent_global_edits_with_direction(
                    stylespace_latent, editing_degrees, editing_name
                ))
        elif editing_name in ["beard_sc"]:
            stylespace_latent = get_stylespace_from_w(original_latent, self.method.decoder)
            edited_latents = (
                self.latent_editor.get_single_styleclip_latent_beard_edits_with_direction(
                    stylespace_latent, editing_degrees, editing_name
                ))
        else:
            raise ValueError(f'Edit name {editing_name} is not available')
        return edited_latents

    def _setup_latent_editor(self):
        self.latent_editor = LatentEditorWrapper(self.config.exp.domain)

    def _setup_device(self):
        config_device = self.config.model["device"].lower()

        if config_device == "cpu":
            device = "cpu"
        elif config_device.isdigit():
            device = "cuda:{}".format(config_device)
        elif config_device.startswith("cuda:"):
            device = config_device
        else:
            raise ValueError("Incorrect Device Type")

        try:
            torch.randn(1).to(device)
            print("Device: {}".format(device))
        except Exception as e:
            print("Could not use device {}, {}".format(device, e))
            print("Set device to CPU")
            device = "cpu"

        self.device = torch.device(device)

    def _setup_method(self):
        method_name = self.config.model.method
        self.method = methods_registry[method_name](
            checkpoint_path=self.config.model.checkpoint_path,
            **self.config.methods_args[method_name],
        ).to(self.device)

    # def _setup_metrics(self):
    #     metrics_names = self.config.train.val_metrics

    #     self.metrics = []
    #     for metric_name in metrics_names:
    #         if hasattr(self.config.metrics, metric_name):
    #             self.metrics.append(
    #                 metrics_registry[metric_name](
    #                     **getattr(self.config.metrics, metric_name)
    #                 )
    #             )
    #         else:
    #             self.metrics.append(metrics_registry[metric_name]())

    # def _setup_wandb(self, options_path=""):
    #     if options_path != "":
    #         with open(options_path, "r") as f:
    #             options = json.load(f)
    #         self.start_step = options["start_step"]
    #     if not self.config.exp.wandb:
    #         return

    #     if options_path == "":
    #         config_for_logger = omegaconf.OmegaConf.to_container(self.config)
    #         self.wandb_args = {
    #             "id": wandb.util.generate_id(),
    #             "project": self.config.exp.project,
    #             "name": self.config.exp.name,
    #             "notes": self.config.exp.notes,
    #             "config": config_for_logger,
    #         }
    #         wandb.init(**self.wandb_args, resume="allow")
    #         run_dir = wandb.run.dir
    #         code = wandb.Artifact("project-source", type="code")
    #         for path in glob.glob("**/*.py", recursive=True):
    #             if not path.startswith("wandb"):
    #                 if os.path.basename(path) != path:
    #                     code.add_dir(
    #                         os.path.dirname(path), name=os.path.dirname(path)
    #                     )
    #                 else:
    #                     code.add_file(os.path.basename(path), name=path)
    #         wandb.run.log_artifact(code)
    #     else:
    #         print(f"Resume training from {self.config.train.resume_path}")
    #         with open(options_path, "r") as f:
    #             options = json.load(f)
    #         self.wandb_args = {
    #             "id": options['id'],
    #             "project": options['project'],
    #             "name": options['name'],
    #             "notes": options['notes'],
    #             "config": options['config'],
    #         }
    #         wandb.init(resume=True, **self.wandb_args)
