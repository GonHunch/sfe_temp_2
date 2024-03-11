import torch
import torch.utils.data
import torch.distributions
import torchvision

from utils.class_registry import ClassRegistry

loaders = ClassRegistry()


@loaders.add_to_registry("classic")
class ClassicLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=True,
        pin_memory=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            multiprocessing_context="fork" if num_workers > 0 else None,
            **kwargs,
        )


@loaders.add_to_registry("infinite")
class InfiniteLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=True,
        device=None,
        pin_memory=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            multiprocessing_context="fork" if num_workers > 0 else None,
            **kwargs,
        )
        self.infinite = infinite
        self.device = device
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            if self.infinite:
                self.dataset_iterator = super().__iter__()
                x = next(self.dataset_iterator)
            else:
                raise
        if self.device is not None:
            x = utils.move_to_device(x, self.device)
        return x
