from abc import abstractmethod
import torchvision.transforms as transforms
from utils.class_registry import ClassRegistry

transforms_registry = ClassRegistry()


class TransformsConfig(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_transforms(self):
        pass


@transforms_registry.add_to_registry(name="encoder")
class EncodeTransforms(TransformsConfig):
    def __init__(self):
        super(EncodeTransforms, self).__init__()

    def get_transforms(self):
        transforms_dict = {
            "transform_gt_train": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_source": None,
            "transform_test": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_inference": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
        }
        return transforms_dict


@transforms_registry.add_to_registry(name="hyperinv")
class HyperinvTransforms(TransformsConfig):
    def __init__(self):
        super(HyperinvTransforms, self).__init__()
        self.image_size = (1024, 1024)

    def get_transforms(self):
        transforms_dict = {
            "transform_gt_train": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_source": None,
            "transform_test": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_inference": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
        }
        return transforms_dict


@transforms_registry.add_to_registry(name="cars")
class CarsEncodeTransforms(TransformsConfig):
    def __init__(self):
        super(CarsEncodeTransforms, self).__init__()

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


@transforms_registry.add_to_registry(name="cars_full")
class CarsEncodeTransforms(TransformsConfig):
    def __init__(self):
        super(CarsEncodeTransforms, self).__init__()

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((384, 512)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.Resize((384, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((384, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict
