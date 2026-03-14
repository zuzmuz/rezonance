from pathlib import Path
import torch
import matplotlib.pyplot as plt
from src.logger import logger
from src.utils import freq_from_pitch
from src.dataset import NSynthDataset


def run(*args, **kwargs):
    
    folder = Path("data", "nsynth-test")
    dataset = NSynthDataset(folder, 16_000, 1024)

    # logger.debug(f"dataset length {len(dataset)}")
    #
    # 
    # signal = dataset[1][0][0]
    #
    # for i in range(5, 11):
    #     plt.subplot(6, 1, i-4)
    #     plt.plot(signal[(i*1024):(i+1)*1024])
    # plt.show()
