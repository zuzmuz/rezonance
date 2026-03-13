import torch
import numpy as np
import matplotlib.pyplot as plt

from src.waveform import Instrument, WaveformSynth
from src.dataset import RandomTimbralDataSet
from src.utils import (
    get_rank_of_pitch,
    freq_from_pitch,
    get_pitch_of_rank,
)


def run(*args, **kwargs):
    sample_rate = 1

    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    instruments = [
        Instrument.saw(
            buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
        ),
        Instrument.square(
            buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
        ),
    ]
   
    for instrument in instruments:
        plt.figure()

        signals = instrument.generate(
            torch.linspace(50, 60, 2, dtype=torch.float32), per_pitch=1
        )


        len_ = len(signals)
        lines = len_

        for idx, element in enumerate(signals):  # type: ignore
            plt.subplot(lines, 1, idx + 1)
            plt.plot(element.cpu().detach().numpy())

    plt.show()
