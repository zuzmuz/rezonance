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

    noises = [
        (Noise.white(0.1), "white 0.1"),
        (Noise.pink(0.2), "pink 0.2"),
        (Noise.brown, 0.3, "brown 0.3"),
        (Noise.blue, 0.8, "blue 0.8"),
    ]

    waveform = waveform_synth.gen_single(60, 0)  # type: ignore

    plt.figure()
    for idx, (noise_func, noise_power, title) in enumerate(noises):
        plt.subplot(len(noises), 1, idx + 1)
        plt.title(title)
        noise = noise_func(noise_synth.gaussian)
        print(f'power of noise "{title}": {np.std(noise)}')
        plt.plot(
            waveform + noise_power * noise_func(noise_synth.gaussian),
            linewidth=0.5,
        )

    plt.show()
