import torch
import numpy as np
import matplotlib.pyplot as plt

from src.waveform import Timbre, WaveformSynth
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
    synth = WaveformSynth(
        sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
    )

    pitches = torch.linspace(50, 60, 2, dtype=torch.float32)
    freqs = freq_from_pitch(pitches, A4=A4)
    ranks = get_rank_of_pitch(
        pitches, sample_rate=sample_rate, A4=A4
    ).floor()
    alpha = 1
    signals = []
    for rank, pitch in zip(ranks, pitches):
        multipliers = torch.arange(1, int(rank + 1))
        freqs = freq_from_pitch(pitch, A4=A4) * multipliers
        powers = 1 / (multipliers**alpha)
        phases = torch.zeros_like(powers)

        sig = synth.gen_poly(
            freqs.unsqueeze(0),
            phases.unsqueeze(0),
            powers.unsqueeze(0),
        )[0]
        signals.append(sig)

    plt.figure()

    len_ = len(signals)
    lines = len_

    for idx, element in enumerate(signals):  # type: ignore
        plt.subplot(lines, 1, idx + 1)
        plt.plot(element.cpu().detach().numpy())

    plt.show()
