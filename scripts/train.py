import torch
import numpy as np
import matplotlib.pyplot as plt

from src.training import SineWaveformDataset, MonophonicModel, Trainer
from src.waveform import WaveformSynth, Noise


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = np.float32(16_000)
    buffer_size = np.int16(1024)
    A4 = np.float32(440)

    dataset = SineWaveformDataset(
        1000,
        100,
        [
            # TODO: consider smart noise combinations
            (Noise.white, 0.01),
            (Noise.white, 0.02),
            (Noise.white, 0.03),
            (Noise.white, 0.05),
            (Noise.white, 0.1),
            (Noise.white, 0.2),
            (Noise.pink, 0.05),
            (Noise.pink, 0.1),
            (Noise.pink, 0.2),
            (Noise.brown, 0.05),
            (Noise.brown, 0.1),
            (Noise.brown, 0.2),
            (Noise.brown, 0.3),
            (Noise.brown, 0.4),
            (Noise.blue, 0.05),
            (Noise.blue, 0.1),
            (Noise.blue, 0.2),
            (Noise.blue, 0.3),
            (Noise.blue, 0.4),
            (Noise.violet, 0.05),
            (Noise.violet, 0.1),
            (Noise.violet, 0.2),
            (Noise.violet, 0.5),
        ],
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )

    trainer = Trainer(MonophonicModel(buffer_size))

    history = trainer.train(1000, dataset)
    # plt.figure()
    # plt.title('history')
    # plt.plot(history)
    plt.show()
    trainer.save_model("monophonic_model.pth")
