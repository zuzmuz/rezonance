from pathlib import Path
import torch
import matplotlib.pyplot as plt

from rezonance.waveform import Instrument


def main():
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    torch.manual_seed(3)

    instruments = [
        ("Sawtooth", Instrument.saw(
            buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
        )),
        ("Square", Instrument.square(
            buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
        )),
        ("Triangle", Instrument.triangle(
            buffer_size=buffer_size, sample_rate=sample_rate, A4=A4
        )),
    ]

    plt.figure()
    lines = 3 * 2
    for instr_idx, (title, instrument) in enumerate(instruments):

        signals = instrument.generate(
            torch.tensor([60]),
            per_pitch=2,
        )


        for idx, element in enumerate(signals):  # type: ignore
            plt.subplot(lines, 1, instr_idx*2 + idx + 1)
            plt.plot(element.cpu().detach().numpy())
            plt.title(title)

    # plt.show()
    plt.savefig(Path("figures", "known_signals.png"))


if __name__ == "__main__":
    main()
