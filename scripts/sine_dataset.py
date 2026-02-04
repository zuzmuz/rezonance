import matplotlib.pyplot as plt

from src.nn import SineWaveformDataset


def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = SineWaveformDataset(
        5,
        3,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
        min_pitch=60,
        max_pitch=70,
    )

    plt.figure()

    for idx, element in enumerate(dataset): # type: ignore
        plt.subplot(5, 3, idx + 1)
        plt.plot(element[0].cpu().detach().numpy(), linewidth=0.5)

    plt.show()
