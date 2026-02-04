import torch
import matplotlib.pyplot as plt
from src.utils import freq_from_pitch
from src.waveform import WaveformSynth


def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    waveform_synth = WaveformSynth(
        sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
    )

    data = torch.tensor(
        [
            [[57, 0, 0.6], [69, 0.2, 0.3]],
            [[57, 0, 0.6], [69, 0.2, 0.0]],
            [[57, 0, 0.0], [69, 0.2, 0.3]],
        ]
    )

    data[:, :, 0] = freq_from_pitch(data[:, :, 0], A4=A4)

    waveform = waveform_synth.gen_poly(data)

    plt.figure()
    plt.plot(waveform[0].cpu().detach().numpy(), linewidth=0.5)
    plt.plot(waveform[1].cpu().detach().numpy(), linewidth=0.5)
    plt.plot(waveform[2].cpu().detach().numpy(), linewidth=0.5)
    plt.show()
