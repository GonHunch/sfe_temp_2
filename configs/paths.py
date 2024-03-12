from dataclasses import dataclass, asdict, fields


models_dir = "../stylegan-encoders/"


@dataclass
class DefaultPathsClass:
    psp_path: str = models_dir + "pretrained_models/psp_ffhq_encode.pt"
    e4e_path: str = models_dir + "pretrained_models/e4e_ffhq_encode.pt"
    farl_path:str = models_dir + "pretrained_models/face_parsing.farl.lapa.main_ema_136500_jit191.pt"
    mobile_net_pth: str = models_dir + "pretrained_models/mobilenet0.25_Final.pth"
    ir_se50_path: str = models_dir + "pretrained_models/model_ir_se50.pth"
    stylegan_weights: str = models_dir + "pretrained_models/stylegan2-ffhq-config-f.pt"
    stylegan_car_weights: str = models_dir + "pretrained_models/stylegan2-car-config-f-new.pkl"
    stylegan_weights_pkl: str = models_dir + "pretrained_models/stylegan2-ffhq-config-f.pkl"
    arcface_model_path: str = models_dir + "pretrained_models/iresnet50-7f187506.pth"
    moco: str = models_dir + "pretrained_models/moco_v2_800ep_pretrain.pt"
    curricular_face_path: str = models_dir + "pretrained_models/CurricularFace_Backbone.pth"
    mtcnn: str = models_dir + "pretrained_models/mtcnn"
    landmark: str = models_dir + "pretrained_models/79999_iter.pth"

    def __iter__(self):
      for field in fields(self):
          yield field.name, getattr(self, field.name)


DefaultPaths = DefaultPathsClass()
