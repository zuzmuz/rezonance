import torch
import numpy as np
import matplotlib.pyplot as plt

from src.training import SineWaveformDataset, MonophonicModel, Trainer
from src.waveform import WaveformSynth

def run(*args, verbose: bool = False, **kwargs):
    sample_rate = np.float32(16_000)
    buffer_size = np.int16(1024)
    A4 = np.float32(440)

    dataset = SineWaveformDataset(
        1000,
        100,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )

    if verbose:
        plt.figure()
        for idx in range(16 * 600, 16 * 600 + 6):
            waveform, pitch = dataset[idx]
            plt.subplot(3, 2, idx + 1 - 16 * 600)
            plt.plot(
                waveform.detach().cpu().numpy(),
                label=f"{pitch:.2f}",
            )
            plt.legend()

    trainer = Trainer(MonophonicModel(buffer_size))

    history = trainer.train(1000, dataset)
    # plt.figure()
    # plt.title('history')
    # plt.plot(history)
    plt.show()
    trainer.save_model("monophonic_model.pth")
