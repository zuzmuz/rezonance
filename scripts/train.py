import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import (
    NoisySineWaveformDataset,
    SineWaveformDataset,
)
from src.nn import (
    LinearModel1,
    Trainer,
)
from src.waveform import WaveformSynth, Noise


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = NoisySineWaveformDataset(
        500,
        50,
        noises=[
            Noise.white(0.05),
            Noise.pink(0.1),
            Noise.brown(0.2),
        ],
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )

    trainer = Trainer(LinearModel1(buffer_size))

    history = trainer.train(6, dataset)
    # plt.figure()
    # plt.title('history')
    # plt.plot(history)
    # plt.show()
    trainer.save_model("monophonic_model.pth")
