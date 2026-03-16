import matplotlib.pyplot as plt

from src.waveform import Instrument
from src.train_dataset import InstrumentSynthDataset


def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    dataset = InstrumentSynthDataset(
        2,
        2,
        instrument=Instrument.sine(
            buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
        ),
        sample_rate=sample_rate,
        A4=A4,
        min_pitch=60,
        max_pitch=70,
    )

    plt.figure()

    for idx, element in enumerate(dataset):  # type: ignore
        plt.subplot(4, 1, idx + 1)
        plt.plot(element[0].cpu().detach().numpy())

    plt.show()
