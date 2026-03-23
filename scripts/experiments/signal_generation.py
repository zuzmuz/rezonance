"""
Visualize different synthetic waveform generation.
We use the instrumentSynth class to generate signals following a certain formant or timbre.
The timbre is defined by the instrumentSynth class.
"""

import matplotlib.pyplot as plt

import torch

from rezonance.defaults import SAMPLE_RATE, BUFFER_SIZE, A4
from rezonance.audioutils.waveform_generators import Instrument


def main():

    instruments = [
        (
            "sine",
            Instrument.sine(
                buffer_size=BUFFER_SIZE,
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
        ),
        (
            "triangle",
            Instrument.triangle(
                buffer_size=BUFFER_SIZE,
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
        ),
        (
            "squre",
            Instrument.square(
                buffer_size=BUFFER_SIZE,
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
        ),
        (
            "saw",
            Instrument.saw(
                buffer_size=BUFFER_SIZE,
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
        ),
        (
            "random",
            Instrument.random(
                1.5,
                buffer_size=BUFFER_SIZE,
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
        ),
    ]

    per_pitch = 2
    pitches = torch.tensor([40, 70])
    for title, synth in instruments:
        plt.figure()
        signal = synth.generate(pitches, per_pitch=per_pitch)
        plt.subplot(2, 2, 1)
        plt.title(f"{title} pitch ={pitches[0]}")
        plt.plot(signal[0])
        plt.subplot(2, 2, 2)
        plt.title(f"{title} pitch ={pitches[0]}")
        plt.plot(signal[1])
        plt.subplot(2, 2, 3)
        plt.title(f"{title} pitch ={pitches[1]}")
        plt.plot(signal[2])
        plt.subplot(2, 2, 4)
        plt.title(f"{title} pitch ={pitches[1]}")
        plt.plot(signal[3])
    plt.show()


if __name__ == "__main__":
    main()
