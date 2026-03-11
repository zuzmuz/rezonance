import torch
import numpy as np
import matplotlib.pyplot as plt

from src.waveform import Timbre
from src.dataset import RandomTimbralDataSet
from src.utils import get_pitch_rank, freq_from_pitch


def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    pitchs = torch.tensor([30, 100, 150])
    freqs = freq_from_pitch(pitchs, A4=A4)
    ranks = get_pitch_rank(pitchs, sample_rate=sample_rate, A4=A4)

    print(f'{pitchs=}')
    print(f'{freqs=}')
    print(f'{ranks=}')

    # dataset = TimbralWaveformDataset(
    #     3,
    #     timbres=[
    #         Timbre(
    #             torch.tensor(
    #                 [
    #                     [1.0, 0.5, 0.8],
    #                     [2.0, 0.25, 0.4],
    #                     [3.0, 0.5, 0.5],
    #                     [4.0, 0.2, 0.2],
    #                     [5.0, 0.9, 0.1],
    #                     [6.0, 1.5, 0.1],
    #                 ]
    #             ),
    #             sample_rate=sample_rate,
    #         ),
    #     ],
    #     sample_rate=sample_rate,
    #     buffer_size=buffer_size,
    #     A4=A4,
    #     min_pitch=60,
    #     max_pitch=140,
    # )
    #
    # # plt.figure()
    # #
    # # len_ = len(dataset)
    # # lines = len_ / 3
    # #
    # # for idx, element in enumerate(dataset):  # type: ignore
    # #     plt.subplot(lines, 3, idx + 1)
    # #     plt.plot(element[0].cpu().detach().numpy())
    #
    # plt.show()
