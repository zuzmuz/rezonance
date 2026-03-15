import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.logger import logger
from src.train_dataset import RandomTimbralDataset
from src.real_dataset import NSynthDataset
from src.training import (
    FCLinearModel,
    Trainer,
)
from src.waveform import WaveformSynth
from src.noise_generators import Noise


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = RandomTimbralDataset(
        500,
        1000,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )
    
    loss = nn.MSELoss()

    trainer = Trainer(FCLinearModel(buffer_size), nn.MSELoss())
    logger.info("starting training")
    stamp = time.perf_counter()
    history = trainer.train(6, dataset)
    logger.info(f"finished training {time.perf_counter() - stamp}")

    # plt.figure()
    # plt.title('history')
    # plt.plot(history)
    # plt.show()
    # trainer.save_model("monophonic_model.pth")
