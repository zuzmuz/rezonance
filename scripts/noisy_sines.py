import numpy as np
import matplotlib.pyplot as plt

from src.waveform import NoiseSynth, Noise, WaveformSynth


def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    waveform_synth = WaveformSynth(
        sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
    )

    noises: list[tuple[NoiseSynth, str]] = [
        (Noise.white(0.05), "white 0.05"),
        (Noise.pink(0.1), "pink 0.1"),
        (Noise.brown(0.2), "brown 0.2"),
        (Noise.blue(0.3), "blue 0.3"),
        (Noise.violet(0.1) + Noise.brown(0.2), "violet 0.1 + brown 0.2"),
    ]

    waveform = waveform_synth.gen_single(69, 0)

    plt.figure()
    for idx, (synth, title) in enumerate(noises):
        plt.subplot(len(noises), 1, idx + 1)
        plt.title(title)
        noise = synth(buffer_size)
        print(f'power of noise "{title}": {noise.std()}')
        plt.plot(
            (waveform + noise).cpu().detach().numpy(),
            linewidth=0.5,
        )

    plt.show()
