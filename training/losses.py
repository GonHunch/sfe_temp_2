import torch
import clip

from torch import nn
import torch.nn.functional as F
from criteria import id_loss, w_norm, moco_loss, id_vit_loss
from criteria.lpips.lpips import LPIPS
#from criteria.bisinet import BiSeNet
from utils.class_registry import ClassRegistry
from utils.model_utils import stylegan_to_classifier
from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, InterpolationMode
from configs.paths import DefaultPaths
#from dreamsim import dreamsim


losses = ClassRegistry()
mask_losses = ClassRegistry()
crop_losses = ClassRegistry()
adv_losses = ClassRegistry()
disc_losses = ClassRegistry()
other_losses = ClassRegistry()


class LossBuilder:
    def __init__(self, enc_losses_dict, disc_losses_dict, device):
        self.coefs_dict = enc_losses_dict
        self.losses_names = [k for k, v in enc_losses_dict.items() if v > 0]
        self.losses = {}
        self.crop_losses = {}
        self.mask_losses = {}
        self.adv_losses = {}
        self.other_losses = {}
        self.device = device

        for loss in self.losses_names:
            if loss in losses.classes.keys():
                self.losses[loss] = losses[loss]().to(self.device).eval()
            elif loss in mask_losses.classes.keys():
                self.mask_losses[loss] = mask_losses[loss](self.device)
            elif loss in crop_losses.classes.keys():
                self.crop_losses[loss] = crop_losses[loss]().to(self.device).eval()
            elif loss in adv_losses.classes.keys():
                self.adv_losses[loss] = adv_losses[loss]()
            elif loss in other_losses.classes.keys():
                self.other_losses[loss] = other_losses[loss]()
            else:
                raise ValueError(f'Unexepted loss: {loss}')

        self.disc_losses = []
        for loss_name, loss_args in disc_losses_dict.items():
            if loss_args.coef > 0:
                self.disc_losses.append(disc_losses[loss_name](**loss_args))


    def encoder_loss(self, batch_data):
        loss_dict = {}
        global_loss = 0.0

        for loss_name, loss in self.losses.items():
            loss_val = loss(batch_data["y_hat"], batch_data["x"])
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        for loss_name, loss in self.mask_losses.items():
            loss_val = loss(batch_data["y_hat"], batch_data["x"], batch_data["mask"])
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        for loss_name, loss in self.crop_losses.items():
            loss_val = loss(
                batch_data["y_hat"][:, :, 35:223, 32:220],
                batch_data["x"][:, :, 35:223, 32:220],
            )
            assert torch.isfinite(loss_val)
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        for loss_name, loss in self.other_losses.items():
            loss_val = loss(batch_data)
            assert torch.isfinite(loss_val)
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        if batch_data["use_adv_loss"]:
            for loss_name, loss in self.adv_losses.items():
                loss_val = loss(batch_data["fake_preds"])
                global_loss += self.coefs_dict[loss_name] * loss_val
                loss_dict[loss_name] = float(loss_val)

        return global_loss, loss_dict

    def disc_loss(self, D, batch_data):
        disc_losses = {}
        total_disc_loss = torch.tensor([0.], device=self.device)

        for loss in self.disc_losses:
            disc_loss, disc_loss_dict = loss(D, batch_data)
            print(disc_loss_dict)

            total_disc_loss += disc_loss
            disc_losses.update(disc_loss_dict)

        return total_disc_loss, disc_losses



@losses.add_to_registry(name="l2")
class L2Loss(nn.MSELoss):
    pass


@losses.add_to_registry(name="lpips")
class LPIPSLoss(LPIPS):
    pass


@losses.add_to_registry(name="lpips_scale")
class LPIPSScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = LPIPSLoss()

    def forward(self, x, y):
        out = 0
        for res in [256, 128, 64]:
            x_scale = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            y_scale = F.interpolate(y, size=(res, res), mode="bilinear", align_corners=False)
            out += self.loss_fn.forward(x_scale, y_scale).mean()
        return out


@losses.add_to_registry(name="dream")
class DreamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn, _ = dreamsim(pretrained=True, cache_dir='dream_cashe')

    def forward(self, x, y):
        x_scale = F.interpolate((x + 1) / 2, size=(224, 224), mode="bicubic", align_corners=False)
        y_scale = F.interpolate((y + 1) / 2, size=(224, 224), mode="bicubic", align_corners=False)
        out =  self.loss_fn(x_scale, y_scale)
        out = out.mean()
        return out


@other_losses.add_to_registry(name="feat_rec")
class FeatReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        if type((batch["feat_recon"])) == list:
            loss = 0
            for i in range(len(batch["feat_recon"])):
                loss += self.loss_fn(batch["feat_recon"][i], batch["feat_real"][i]).mean()
            return loss
        return self.loss_fn(batch["feat_recon"], batch["feat_real"]).mean()


@other_losses.add_to_registry(name="feat_rec_l1")
class FeatReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, batch):
        if type((batch["feat_recon"])) == list:
            loss = 0
            for i in range(len(batch["feat_recon"])):
                loss += self.loss_fn(batch["feat_recon"][i], batch["feat_real"][i]).mean()
            return loss
        return self.loss_fn(batch["feat_recon"], batch["feat_real"]).mean()
        

@other_losses.add_to_registry(name="mse_reg")
class MseRegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        loss = 0
        for i in range(len(batch["mse_reg_1"])):
            loss += self.loss_fn(batch["mse_reg_1"][i], batch["mse_reg_2"][i]).mean()
        return loss


@other_losses.add_to_registry(name="l2_synt")
class SyntMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        bs = batch['x'].size(0) // 4
        x = torch.cat([batch["x"][bs:2*bs], batch["x"][3*bs:]], dim = 0)
        y = torch.cat([batch["y_hat"][bs:2*bs], batch["y_hat"][3*bs:]], dim = 0)
        return self.loss_fn(x, y).mean() / 2


@other_losses.add_to_registry(name="l2_latent")
class LatentMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.loss_fn(batch["latent"], batch["latent_rec"]).mean()


@losses.add_to_registry(name="landmark")
class Landmark(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.parsing_net = BiSeNet(n_classes=19)
        self.parsing_net.load_state_dict(torch.load(DefaultPaths.landmark))
        self.parsing_net.eval()

    def forward(self, x, y):
        x_1 = stylegan_to_classifier(x, out_size=(512, 512))
        x_2 = stylegan_to_classifier(y, out_size=(512, 512))
        out_1 = self.parsing_net(x_1)
        out_2 = self.parsing_net(x_2)
        parsing_loss = sum(
            [1 - self.cos(out_1[i].flatten(start_dim=1), out_2[i].flatten(start_dim=1)) for i in
             range(len(out_1))])
        return parsing_loss.mean()
        

@losses.add_to_registry(name="clip")
class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.parsing_net, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.preprocess = Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                                  CenterCrop(size=(224, 224)),
                                  Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        self.parsing_net.eval()

    def forward(self, x, y):
        x_1 = self.preprocess(torch.clamp((0.5 * x + 0.5), 0, 1))
        x_2 = self.preprocess(torch.clamp((0.5 * y + 0.5), 0, 1))

        out_1 = self.parsing_net.encode_image(x_1)
        out_2 = self.parsing_net.encode_image(x_2)
        
        parsing_loss = 1 - self.cos(out_1, out_2)
        return parsing_loss.mean()


@losses.add_to_registry(name="id")
class IDLoss(id_loss.IDLoss):
    pass


@losses.add_to_registry(name="id_vit")
class IDVitLoss(id_vit_loss.IDVitLoss):
    pass


@losses.add_to_registry(name="moco")
class MocoLoss(moco_loss.MocoLoss):
    pass


class MaskLoss:
    def __init__(self, inner_loss, device):
        self.fn = inner_loss().to(device).eval()
    
    def fuser(self, x, y, mask):
        mask3 = mask.repeat(1, 3, 1, 1)
        return (1 - mask3) * x + mask3 * y.detach()

    def __call__(self, y_hat, x, mask):
        return self.fn(y_hat, self.fuser(x, y_hat, mask))


@mask_losses.add_to_registry(name="l2mask")
class L2MaskLoss(MaskLoss):
    def __init__(self, device):
        super().__init__(nn.MSELoss, device)


@mask_losses.add_to_registry(name="lpipsmask")
class L2MaskLoss(MaskLoss):
    def __init__(self, device):
        super().__init__(LPIPS, device)


@mask_losses.add_to_registry(name="idmask")
class L2MaskLoss(MaskLoss):
    def __init__(self, device):
        super().__init__(id_loss.IDLoss, device)


@mask_losses.add_to_registry(name="mocomask")
class L2MaskLoss(MaskLoss):
    def __init__(self, device):
        super().__init__(moco_loss.MocoLoss, device)


@crop_losses.add_to_registry(name="l2_crop")
class L2Loss(nn.MSELoss):
    pass


@crop_losses.add_to_registry(name="lpips_crop")
class LPIPSLoss(LPIPS):
    pass


@adv_losses.add_to_registry(name="adv")
class EncoderAdvLoss:
    def __call__(self, fake_preds):
        loss_G_adv = F.softplus(-fake_preds).mean()
        return loss_G_adv


@disc_losses.add_to_registry(name="main")
class AdvLoss:
    def __init__(self, coef=0.0):
        self.coef = coef

    def __call__(self, disc, loss_input):
        real_images = loss_input["x"].detach()
        generated_images = loss_input["y_hat"].detach()
        loss_dict = {}

        fake_preds = disc(generated_images, None)
        real_preds = disc(real_images, None)
        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["disc/main_loss"] = float(loss)

        return loss, loss_dict

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2


@disc_losses.add_to_registry(name="r1")
class R1Loss:
    def __init__(self, coef=0.0, hyper_d_reg_every=16):
        self.coef = coef
        self.hyper_d_reg_every = hyper_d_reg_every

    def __call__(self, disc, loss_input):
        real_images = loss_input["x"]
        step = loss_input["step"]
        if step % self.hyper_d_reg_every != 0:  # use r1 only once per 'hyper_d_reg_every' steps
            return torch.tensor([0.], requires_grad=True, device='cuda'), {}

        real_images.requires_grad = True
        loss_dict = {}

        real_preds = disc(real_images, None)
        real_preds = real_preds.view(real_images.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        r1_loss = self.d_r1_loss(real_preds, real_images)

        loss_D_R1 = self.coef / 2 * r1_loss * self.hyper_d_reg_every + 0 * real_preds[0]
        loss_dict["disc/r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty
