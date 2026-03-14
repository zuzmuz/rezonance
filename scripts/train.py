import logging
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.logger import logger
from src.dataset import (
    NoisySineWaveformDataset,
    RandomTimbralDataSet,
)
from src.training import (
    FCModel,
    Trainer,
)
from src.waveform import WaveformSynth
from src.noise_generators import Noise


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    # dataset = NoisySineWaveformDataset(
    #     500,
    #     50,
    #     noises=[
    #         Noise.white(0.05),
    #         Noise.pink(0.1),
    #         Noise.brown(0.2),
    #     ],
    #     sample_rate=sample_rate,
    #     buffer_size=buffer_size,
    #     A4=A4,
    # )

    dataset = RandomTimbralDataSet(
        500,
        1000,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )

    trainer = Trainer(FCModel(buffer_size))
    logger.info("starting training")
    stamp = time.perf_counter()
    history = trainer.train(6, dataset)
    logger.info(f"finished training {time.perf_counter() - stamp}")

    # plt.figure()
    # plt.title('history')
    # plt.plot(history)
    # plt.show()
    # trainer.save_model("monophonic_model.pth")
