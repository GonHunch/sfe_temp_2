# import os

# from pathlib import Path
# from typing import Optional, List, Tuple, Dict
# from dataclasses import dataclass, field
# from omegaconf import OmegaConf, MISSING
# from utils.class_registry import ClassRegistry
# from models.inverters import inverter_registry
# #from metrics.edit_metrics import editing_metrics
# from metrics.metrics import metrics_registry
# from training.losses import disc_losses
# from training.optimizers import optimizers


# args = ClassRegistry()


# @args.add_to_registry("exp")
# @dataclass
# class ExperimentArgs:
#     config_dir: str = str(Path(__file__).resolve().parent / "configs")
#     config: str = "base_config.yaml"
#     exp_dir: str = "experiments"
#     project: str = "GANLatent"
#     name: str = MISSING
#     seed: int = 1
#     root: str = os.getenv("EXP_ROOT", ".")
#     notes: str = "empty notes"
#     logging: bool = True
#     wandb: bool = True
#     save_outputs: bool = False
#     domain: str = "human_faces"


# @args.add_to_registry("data")
# @dataclass
# class DataArgs:
#     special_dir: str = "./special_pics"
#     inference_dir: str = ""
#     transform: str = "encoder"
#     input_train_dir: str = ""
#     input_val_dir: str = ""


# @args.add_to_registry("inference")
# @dataclass
# class InferenceArgs:
#     inference_runner: str = "base_inference_runner"
#     metrics: List[str] = field(
#         default_factory=lambda: ["msssim", "lpips", "l2"]
#     )
#     editing_metrics: List[str] = field(
#         default_factory=lambda: [])
#     editing_attrs: Dict = field(default_factory=lambda: {})
#     fid_eiditing_map: Dict = field(default_factory=lambda: {})


# @args.add_to_registry("train")
# @dataclass
# class TrainingArgs:
#     coach: str = "base_coach"
#     loss: str = "psp_loss"
#     encoder_optimizer: str = "adam"
#     disc_optimizer: str = ""
#     transform: str = "encoder"
#     resume_path: str = ""
#     val_metrics: List[str] = field(
#         default_factory=lambda: ["msssim", "lpips", "l2", "fid"]
#     )
#     start_step: int = 0
#     steps: int = 300000
#     log_step: int = 5000
#     checkpoint_step: int = 30000
#     val_step: int = 30000
#     save_val_images: bool = False
#     train_dis: bool = False
#     dis_train_start_step: int = 150000
#     bs_used_before_adv_loss: int = 8
#     adv_warmup: int = 1
#     use_synthesis: bool = False
#     masked: bool = False
#     disc_edits: List[str] = field(
#         default_factory=lambda: []
#     )
#     edits_adv: bool = False


# @args.add_to_registry("model")
# @dataclass
# class ModelArgs:
#     inverter: str = "psp"
#     device: str = "0"
#     batch_size: int = 4
#     workers: int = 4
#     checkpoint_path: str = ""


# @args.add_to_registry("encoder_losses")
# @dataclass
# class EncoderLossesArgs:
#     l2: float = 0.0
#     lpips: float = 0.0
#     lpips_scale: float = 0.0
#     id: float = 0.0
#     moco: float = 0.0
#     l2mask: float = 0.0
#     lpipsmask: float = 0.0
#     idmask: float = 0.0
#     mocomask: float = 0.0
#     w_norm: float = 0.0
#     l2_crop: float = 0.0
#     l2_synt: float = 0.0
#     lpips_crop: float = 0.0
#     adv: float = 0.0
#     feat_rec: float = 0.0
#     feat_rec_l1: float = 0.0
#     landmark: float = 0.0
#     l2_latent: float = 0.0
#     mse_reg: float = 0.0
#     clip: float = 0.0
#     dream: float = 0.0
#     id_vit: float = 0.0


# @args.add_to_registry("editings")
# @dataclass
# class EditingArgs:
#     editings_data: Dict = field(default_factory=lambda: {})


# InvertersArgs = inverter_registry.make_dataclass_from_args("InvertersArgs")
# args.add_to_registry("inverters_args")(InvertersArgs)

# MetricsArgs = metrics_registry.make_dataclass_from_args("MetricsArgs")
# args.add_to_registry("metrics")(MetricsArgs)

# # EditMetricsArgs = editing_metrics.make_dataclass_from_args("EditMetricsArgs")
# # args.add_to_registry("edit_metrics")(EditMetricsArgs)

# DiscLossesArgs = disc_losses.make_dataclass_from_args("DiscLossesArgs")
# args.add_to_registry("disc_losses")(DiscLossesArgs)

# OptimizersArgs = optimizers.make_dataclass_from_args("OptimizersArgs")
# args.add_to_registry("optimizers")(OptimizersArgs)


# Args = args.make_dataclass_from_classes("Args")


# def load_config():
#     config = OmegaConf.structured(Args)

#     conf_cli = OmegaConf.from_cli()
#     config.exp.config = conf_cli.exp.config
#     config.exp.config_dir = conf_cli.exp.config_dir

#     config_path = os.path.join(config.exp.config_dir, config.exp.config)
#     conf_file = OmegaConf.load(config_path)
#     config = OmegaConf.merge(config, conf_file)

#     config = OmegaConf.merge(config, conf_cli)

#     return config


# # if __name__ == "__main__":
# #     config_base = OmegaConf.structured(Args)
# #     config_base_path = os.path.join(config_base.exp.config_dir, config_base.exp.config)

# #     conf_cli = OmegaConf.from_cli()

# #     if conf_cli.get("merge_with", False):
# #         print("merged with config: ", conf_cli["merge_with"])
# #         config_base = OmegaConf.merge(
# #             config_base, OmegaConf.load(conf_cli["merge_with"])
# #         )

# #     if not Path(config_base_path).exists():
# #         with open(config_base_path, "w") as fout:
# #             OmegaConf.save(config=config_base, f=fout.name)
