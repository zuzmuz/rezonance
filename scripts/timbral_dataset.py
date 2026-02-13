import torch
import matplotlib.pyplot as plt

from src.waveform import Timbre
from src.dataset import TimbralWaveformDataset

def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = TimbralWaveformDataset(
        2,
        timbres=[
            Timbre(
                torch.tensor([
                    [1.0, 0.5, 0.8],
                    [2.0, 0.25, 0.5],
                ]),
                sample_rate=sample_rate,
            ),
        ],
        min_pitch=60,
        max_pitch=64
    )
