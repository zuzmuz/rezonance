import numpy as np
import matplotlib.pyplot as plt

from src.waveform import NoiseSynth


def run(*args, **kwargs):
    sample_rate = np.float32(16_000)
    buffer_size = np.int16(1024)
    noise_synth = NoiseSynth(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
    )

    plt.figure(figsize=(13, 8))

    noises = [
        ("white guassian", "black", noise_synth.gaussian.white()),
        ("pink guassian", "hotpink", noise_synth.gaussian.pink()),
        ("brown guassian", "brown", noise_synth.gaussian.brown()),
        ("blue guassian", "blue", noise_synth.gaussian.blue()),
        ("violet guassian", "violet", noise_synth.gaussian.violet()),
        # ("white uniform", "black", noise_synth.uniform.white()),
        # ("pink uniform", "hotpink", noise_synth.uniform.pink()),
        # ("brown uniform", "brown", noise_synth.uniform.brown()),
        # ("blue unifiform", "blue", noise_synth.uniform.blue()),
        # ("violet uniform", "violet", noise_synth.uniform.violet()),
    ]

    noises_len = len(noises)

    for idx, (title, color, noise) in enumerate(noises):
        plt.subplot(noises_len, 2, idx * 2 + 1)
        plt.title(title)
        plt.plot(noise, color=color, linewidth=0.5)
        plt.subplot(noises_len, 2, idx * 2 + 2)
        plt.title(f"{title} fft")
        plt.plot(np.abs(np.fft.fft(noise)), color=color, linewidth=0.5)

    # plt.subplot(3, 1, 3)
    # plt.title("Pink Noise")
    # plt.plot(pink_noise)

    plt.tight_layout()
    plt.show()
