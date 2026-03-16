import torch
import matplotlib.pyplot as plt
import importlib
import pkgutil
from argparse import ArgumentParser
from src.logger import logger
from src.utils import current_device


def main():
    torch.set_default_device(current_device)

    logger.info(f"Using device: {torch.get_default_device()}")

    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.autolayout"] = True
    # plt.style.use('dark_background')

    parser = ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=[
            module_name
            for _, module_name, _ in pkgutil.iter_modules(["scripts"])
        ],
        help="Mode to run the application in.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to load the model from (required for validate mode).",
    )
    args = parser.parse_args()

    module = importlib.import_module(f"scripts.{args.mode}")
    module.run(verbose=args.verbose)


if __name__ == "__main__":
    main()
