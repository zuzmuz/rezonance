from pathlib import Path
import torch
import matplotlib.pyplot as plt
from src.logger import logger
from src.utils import freq_from_pitch
from src.dataset import NSynthDataset


def run(*args, **kwargs):
    
    folder = Path("data", "nsynth-test")
    dataset = NSynthDataset(folder, 16_000, 1024)

    logger.debug(f"dataset length {len(dataset)}")

    logger.debug(f"first item {dataset[0][1]}")
