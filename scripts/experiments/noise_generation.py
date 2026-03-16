import numpy as np
import matplotlib.pyplot as plt

from rezonance.noise_generators import Noise, NoiseSynth

def main():
    buffer_size = 1024

    plt.figure()

    noises: list[tuple[str, str, NoiseSynth]] = [
        ("white", "black", Noise.white(1.0)),
        ("pink", "hotpink", Noise.pink(1.0)),
        ("brown", "brown", Noise.brown(1.0)),
        ("blue", "blue", Noise.blue(1.0)),
        ("violet", "violet", Noise.violet(1.0)),
        (
            "brown + violet",
            "green",
            Noise.brown(1.4) + Noise.violet(0.3),
        ),
    ]

    noises_len = len(noises)

    for idx, (title, color, synth) in enumerate(noises):
        plt.subplot(noises_len, 2, idx * 2 + 1)
        plt.title(title)
        noise = synth.generate(buffer_size).cpu().detach().numpy()
        plt.plot(noise, color=color, linewidth=0.5)
        plt.subplot(noises_len, 2, idx * 2 + 2)
        plt.title(f"{title} fft")
        plt.plot(
            np.abs(np.fft.fft(noise)), color=color, linewidth=0.5
        )

    plt.show()


if __name__ == '__main__':
    main()
