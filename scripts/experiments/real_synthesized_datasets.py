"""
This file compares one sample from a real dataset,
one that contains files of real recorded instrument and one that is synthetically generated.
The goal is to see the similarities and make sure that our synthetic dataset corresponds to reality
"""
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from rezonance.logger import logger
from rezonance.datasets.real_dataset import FileDataset
from rezonance.audioutils.waveform_generators import Instrument


def main():
    buffer_size = 1024
    sample_rate = 16_000
    A4 = 440

    torch.manual_seed(40)

    folder = Path("data", "nsynth-test")
    dataset = FileDataset(folder, sample_rate, buffer_size, 5)

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
    synth_signal /= synth_signal.std() # normalizing

    plt.subplot(2, 1, 1)
    plt.plot(real_signal)
    plt.title(f"real {pitch=}")

    plt.subplot(2, 1, 2)
    plt.plot(synth_signal[0])
    plt.title(f"synth {pitch=}")
    plt.savefig(Path("figures", "real_vs_synth.png"))


if __name__ == "__main__":
    main()
