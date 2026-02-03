import torch
import numpy as np
import matplotlib.pyplot as plt

from src.nn import SineWaveformDataset, MonophonicModel, Trainer
from src.waveform import WaveformSynth, Noise


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = SineWaveformDataset(
        500,
        50,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )

    trainer = Trainer(MonophonicModel(buffer_size))

    history = trainer.train(50, dataset)
    # plt.figure()
    # plt.title('history')
    # plt.plot(history)
    # plt.show()
    trainer.save_model("monophonic_model.pth")
