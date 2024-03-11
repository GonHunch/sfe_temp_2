import random
import sys

# sys.path = ['', '/home/aalanov/.conda/envs/a300/lib/python310.zip', '/home/aalanov/.conda/envs/a300/lib/python3.10', '/home/aalanov/.conda/envs/a300/lib/python3.10/lib-dynload', '/home/aalanov/.local/lib/python3.10/site-packages', '/home/aalanov/.conda/envs/a300/lib/python3.10/site-packages']
sys.path =  ['.', '..'] + sys.path

import torch
from arguments import training_arguments

from pathlib import Path
from omegaconf import OmegaConf

# sys.path.append(".")
# sys.path.append("..")

from runners.training_runners import training_runners
from utils.common_utils import printer, setup_seed


if __name__ == "__main__":
    config = training_arguments.load_config()
    setup_seed(config.exp.seed)

    printer(config)

    trainer = training_runners[config.train.train_runner](config)
    trainer.setup()
    trainer.run()
