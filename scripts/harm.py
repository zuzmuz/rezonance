import torch
import matplotlib.pyplot as plt
from src.waveform import WaveformSynth

def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    waveform_synth = WaveformSynth(
        sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
    )
    
    waveform = waveform_synth.gen_poly(
        torch.tensor([
            [[69, 0, 0.8], [57, 0, 0.4]]     
        ])
    )

    plt.figure()
    plt.plot(waveform[0].cpu().detach().numpy(), linewidth=0.5)
    plt.show()

