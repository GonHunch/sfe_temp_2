import random
import sys

sys.path =  ['.', '..'] + sys.path

import torch
print(torch.cuda.is_available())
print(torch.__version__)

from arguments import inference_arguments
from runners.inference_runners import inference_runner_registry
from utils.common_utils import printer, setup_seed


def run_inference(config):
    inference_runner = inference_runner_registry[config.inference.inference_runner](
        config
    )
    inference_runner.setup()
    inference_runner.run()


if __name__ == "__main__":
    config = inference_arguments.load_config()
    setup_seed(config.exp.seed)

    printer(config)

    run_inference(config)
