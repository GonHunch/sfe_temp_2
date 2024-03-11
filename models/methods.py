import math
import sys
import pickle
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision.transforms.functional import resize as resize_func
from torchvision.transforms import GaussianBlur, RandomHorizontalFlip, RandomAffine, RandomPerspective
from torchvision.transforms.functional import affine
from torchvision.transforms import GaussianBlur
from models.psp.encoders import psp_encoders
#from models.e4e.encoders import psp_encoders as e4e_psp_encoders
from models.psp.stylegan2.model import Generator, Discriminator
from models.hyperinverter.stylegan2_ada import Discriminator as Discriminator_hyperinv
from utils.class_registry import ClassRegistry
from utils.common_utils import get_keys
from utils.model_utils import get_stylespace_from_w, toogle_grad
from configs.paths import DefaultPaths
from argparse import Namespace
from training import loggers
from training.loggers import BaseTimer


sys.path.append("./utils")
methods_registry = ClassRegistry()


@methods_registry.add_to_registry("psp", stop_args=("self", "checkpoint_path"))
class pSp(nn.Module):
    def __init__(
        self,
        output_size=1024,
        encoder_type="GradualStyleEncoder",
        input_nc=3,
        paths=DefaultPaths,
        label_nc=0,
        start_from_latent_avg=True,
        device="cuda:0",
        latent_mask="",
        resize_outputs=False,
        add_encoder_pth=None,
        mix_alpha=None,
        checkpoint_path=None,
    ):
        super(pSp, self).__init__()
        self.opts = {
            "output_size": output_size,
            "encoder_type": encoder_type,
            "input_nc": input_nc,
            "label_nc": label_nc,
            "learn_in_w": False,
            "start_from_latent_avg": start_from_latent_avg,
            "device": device,
            "checkpoint_path": checkpoint_path,
            "use_stylespace": False,
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2

        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        self.discriminator = Discriminator(1024)

        self.load_weights()


    def set_encoder(self):
        if self.opts.encoder_type == "GradualStyleEncoder":
            encoder = psp_encoders.GradualStyleEncoder(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "BackboneEncoderUsingLastLayerIntoW":
            self.opts.learn_in_w = True
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(
                50, "ir_se", self.opts
            )
        elif self.opts.encoder_type == "BackboneEncoderUsingLastLayerIntoWPlus":
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(
                50, "ir_se", self.opts
            )
        else:
            raise Exception("{} is not a valid encoders".format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading pSp from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, "decoder"), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print("Loading encoders weights from irse50!")
            encoder_ckpt = torch.load(self.opts.ir_se50_path)
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {
                    k: v for k, v in encoder_ckpt.items() if "input_layer" not in k
                }
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print("Loading decoder weights from pretrained!")
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            self.discriminator.load_state_dict(ckpt["d"], strict=True)
            self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

        with torch.no_grad():
        	a = self.discriminator(torch.rand(8, 3, 1024, 1024))
        print(type(a))
        print(a.shape)

    def forward(self, x, return_latents=False):
        codes = self.encoder(x)
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        images, result_latent = self.decoder(
            [codes],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )

        if return_latents:
            return images, result_latent
        else:
            return images


    def __load_latent_avg(self, ckpt, repeat=None):
        if "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None


@methods_registry.add_to_registry("e4e", stop_args=("self", "checkpoint_path"))
class e4epSp(nn.Module):
    def __init__(
        self,
        output_size=1024,
        encoder_type="Encoder4Editing",
        input_nc=3,
        paths=DefaultPaths,
        label_nc=0,
        learn_in_w=False,
        start_from_latent_avg=True,
        device="cuda:0",
        latent_mask="",
        resize_outputs=False,
        mix_alpha=None,
        checkpoint_path=None,
        use_stylespace=False,
    ):
        super(e4epSp, self).__init__()
        self.opts = {
            "output_size": output_size,
            "encoder_type": encoder_type,
            "input_nc": input_nc,
            "label_nc": label_nc,
            "learn_in_w": learn_in_w,
            "start_from_latent_avg": start_from_latent_avg,
            "device": device,
            "stylegan_size": output_size,
            "checkpoint_path": checkpoint_path,
            "use_stylespace" : use_stylespace
        }
        self.use_stylespace = use_stylespace
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256)) if self.opts.stylegan_size == 1024 else torch.nn.AdaptiveAvgPool2d((192, 256))
        # Load weights if needed
        self.load_weights()
        self.model_names = ["encoder", "decoder"]

    def set_encoder(self):
        if self.opts.encoder_type == "GradualStyleEncoder":
            encoder = e4e_psp_encoders.GradualStyleEncoder(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "Encoder4Editing":
            encoder = e4e_psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "SingleStyleCodeEncoder":
            encoder = e4e_psp_encoders.BackboneEncoderUsingLastLayerIntoW(
                50, "ir_se", self.opts
            )
        else:
            raise Exception("{} is not a valid encoders".format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print(
                "Loading e4e over the pSp framework from checkpoint: {}".format(
                    self.opts.checkpoint_path
                )
            )
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, "decoder"), strict=True)
            self.__load_latent_avg(ckpt)

        else:
            print("Loading encoders weights from irse50!")
            encoder_ckpt = torch.load(self.opts.ir_se50_path)
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print("Loading decoder weights from pretrained!")
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def forward(
        self,
        x,
        resize=True,
        latent_mask=None,
        input_code=False,
        randomize_noise=True,
        inject_latent=None,
        return_latents=False,
        alpha=None,
    ):
        if input_code:
            codes = x
        else:
            if x.size(2) > 256:
                x = self.face_pool(x)
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = (
                        codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                    )
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = (
                            alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                        )
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder(
            [codes],
            input_is_latent=input_is_latent,
            randomize_noise=randomize_noise,
            return_latents=return_latents,
        )
        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                self.latent_avg = self.decoder.mean_latent(10000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)


@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path"))
class FSEFull(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None,
                 inverter_pth=None):
        super(FSEFull, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.inverter_pth = inverter_pth

        self.encoder = self.set_encoder()

        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.decoder.eval()
        self.latent_avg = None

        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        # We used the hyperinverter discriminator since it has a cars checkpoint
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator_hyperinv(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)


    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading Encoder from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            self.latent_avg = ckpt['latent_avg'].to(self.device)

            print("Loading decoder from", self.opts.stylegan_weights)
        else:
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            print("Loading decoder from", self.opts.stylegan_weights)
            self.latent_avg = ckpt['latent_avg'].to(self.device)

            ckpt = torch.load(self.inverter_pth, map_location="cpu")
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)

    def set_encoder(self):
        fs_backbone = psp_encoders.FSLikeBackbone(opts=self.opts, n_styles=18)
        fuser = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18) #nn.ModuleList([fs_backbone, fuser])

        if self.opts.checkpoint_path == "":
	        ckpt = torch.load(self.inverter_pth, map_location="cpu")
	        inverter.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
        else:
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            inverter.load_state_dict(get_keys(ckpt, "inverter"), strict=True)

        # self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18)
        # self.inverter.fs_backbone = inverter[0]
        # self.inverter.fuser = inverter[1]
        self.inverter = inverter

        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        ckpt = torch.load(self.opts.e4e_path, map_location="cpu")
        self.e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
        self.e4e_encoder = self.e4e_encoder.eval().to(self.device)
        toogle_grad(self.e4e_encoder, False)

        feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        return feat_editor  # trainable part
    
    def forward(self,
                x,
                return_latents=False,
                n_iter=1e5,
                mask=None):
        
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            w_recon = w_recon + self.latent_avg
                    
            _, w_feats = self.decoder([w_recon],
                                       input_is_latent=True,
                                       return_features=True,
                                       is_stylespace=False,
                                       randomize_noise=False,
                                       early_stop=64)

            w_feat = w_feats[9]  # bs x 512 x 64 x 64 
            
            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            delta = torch.zeros_like(fused_feat)  # inversion case

        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.decoder([w_recon],
                                 input_is_latent=True,
                                 return_features=True,
                                 features_in=feats,
                                 feature_scale=min(1.0, 0.0001 * n_iter),
                                 is_stylespace=False,
                                 randomize_noise=False)
        
        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images


@methods_registry.add_to_registry("fse_inverter", stop_args=("self", "checkpoint_path"))
class FSEInverter(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None):
        super(FSEInverter, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.encoder = self.set_encoder()

        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.decoder.eval()
        self.latent_avg = None

        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        # We used the hyperinverter discriminator since it has a cars checkpoint
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator_hyperinv(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)


    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading Encoder from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            self.latent_avg = ckpt['latent_avg'].to(self.device)

            print("Loading decoder from", self.opts.stylegan_weights)
        else:
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            print("Loading decoder from", self.opts.stylegan_weights)
            self.latent_avg = ckpt['latent_avg'].to(self.device)

    def set_encoder(self):
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18)
        return inverter  # trainable part
    
    def forward(self,
                x,
                return_latents=False,
                n_iter=1e5,
                mask=None):
    
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        w_recon, predicted_feat = self.encoder.fs_backbone(x)
        w_recon = w_recon + self.latent_avg
                
        _, w_feats = self.decoder([w_recon],
                                   input_is_latent=True,
                                   return_features=True,
                                   is_stylespace=False,
                                   randomize_noise=False,
                                   early_stop=64)

        w_feat = w_feats[9]  # bs x 512 x 64 x 64 
        
        fused_feat = self.encoder.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        feats = [None] * 9 + [fused_feat] + [None] * (17 - 9)

        images, _ = self.decoder([w_recon],
                                 input_is_latent=True,
                                 return_features=True,
                                 features_in=feats,
                                 feature_scale=min(1.0, 0.0001 * n_iter),
                                 is_stylespace=False,
                                 randomize_noise=False)
        
        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                w_feat = w_feat.cpu()
            return images, w_recon, fused_feat, w_feat
        return images
