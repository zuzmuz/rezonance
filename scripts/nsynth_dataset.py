from pathlib import Path
import torch
import matplotlib.pyplot as plt
from src.logger import logger
from src.utils import freq_from_pitch
from src.real_dataset import NSynthDataset
from src.waveform import Instrument


def run(*args, **kwargs):

    buffer_size = 1024
    sample_rate = 16_000
    A4 = 440

    folder = Path("data", "nsynth-test")
    dataset = NSynthDataset(folder, sample_rate, buffer_size, 5)

    logger.debug(f"dataset length {len(dataset)}")

    real_signal, pitch = dataset[500]

    instrument = Instrument.random(
        1.5, buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
    )

    plt.figure()

    pitch = torch.tensor([40])
    synth_signal = instrument.generate(
        pitch,
        per_pitch=1,
    )

    plt.subplot(2, 1, 1)
    plt.plot(real_signal)
    plt.title(f"real {pitch=}")

    plt.subplot(2, 1, 2)
    plt.plot(synth_signal[0])
    plt.title(f"synth {pitch=}")
    plt.show()
