"""
Visulize different data augmentation technic relevant to the application
"""

import torch
import matplotlib.pyplot as plt

from rezonance.defaults import A4, BUFFER_SIZE, SAMPLE_RATE
from rezonance.audioutils.noise_generators import Noise
from rezonance.audioutils.waveform_generators import Instrument
from rezonance import transforms


def main():

    torch.manual_seed(2)

    instrument = Instrument.random(
        1.5, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE, A4=A4
    )

    signal = instrument.generate(torch.tensor([50]), per_pitch=1)[0]

    augmentations = [
        (
            "noise",
            transforms.noise(Noise.brown(0.1) + Noise.violet(0.05)),
        ),
        ("masking", transforms.mask(100, 0)),
        (
            "scaling",
            transforms.scaling(0.5, 1.2, BUFFER_SIZE),
        ),
    ]

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.plot(signal)
    plt.title("Original Signal")

    for idx, (title, augmentation) in enumerate(augmentations):
        plt.subplot(4, 1, idx + 2)
        plt.plot(augmentation(signal.clone()))
        plt.title(title)

    plt.show()


if __name__ == "__main__":
    main()
