import matplotlib.pyplot as plt

from rezonance.noise_generators import Noise
from rezonance.dataset import NoisySineWaveformDataset

def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = NoisySineWaveformDataset(
        2,
        2,
        noises=[
            Noise.white(0.05),
            Noise.white(0.1),
            Noise.white(0.2),
            Noise.white(0.3),
            Noise.brown(0.2),
            Noise.pink(0.3),
            Noise.pink(0.2),
            Noise.pink(0.1),
            Noise.pink(0.05),
            Noise.pink(0.01),
            Noise.violet(0.2) + Noise.brown(0.2),
            Noise.blue(0.1) + Noise.brown(0.3),
            Noise.blue(0.3) + Noise.brown(0.05),
        ],
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
        min_pitch=60,
        max_pitch=70,
    )

    plt.figure()

    for idx, element in enumerate(dataset): # type: ignore
        plt.subplot(4, 3, idx + 1)
        plt.plot(element[0].cpu().detach().numpy(), linewidth=0.5)

    plt.show()
