from dataclasses import dataclass, asdict


@dataclass
class DefaultPathsClass:
    psp_path: str = "../stylegan-encoders/pretrained_models/psp_ffhq_encode.pt"
    e4e_path: str = "../stylegan-encoders/pretrained_models/e4e_ffhq_encode.pt"
    farl_path:str = "../stylegan-encoders/pretrained_models/face_parsing.farl.lapa.main_ema_136500_jit191.pt"
    mobile_net_pth: str = "../stylegan-encoders/pretrained_models/mobilenet0.25_Final.pth"
    ir_se50_path: str = "../stylegan-encoders/pretrained_models/model_ir_se50.pth"
    stylegan_weights: str = "../stylegan-encoders/pretrained_models/stylegan2-ffhq-config-f.pt"
    stylegan_car_weights: str = "../stylegan-encoders/pretrained_models/stylegan2-car-config-f-new.pkl"
    stylegan_weights_pkl: str = (
        "../stylegan-encoders/pretrained_models/stylegan2-ffhq-config-f.pkl"
    )
    arcface_model_path: str = "../stylegan-encoders/pretrained_models/iresnet50-7f187506.pth"
    moco: str = "../stylegan-encoders/pretrained_models/moco_v2_800ep_pretrain.pt"
    curricular_face_path: str = "../stylegan-encoders/pretrained_models/CurricularFace_Backbone.pth"
    mtcnn: str = "../stylegan-encoders/pretrained_models/mtcnn"
    landmark: str = "../stylegan-encoders/pretrained_models/79999_iter.pth"


DefaultPaths = DefaultPathsClass()
