from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.training import SineWaveformDataset, MonophonicModel, Trainer
from src.waveform import WaveformSynth


class Modes:
    @staticmethod
    def test_generation(*args, **kwargs):
        sample_rate = np.float32(16_000)
        buffer_size = np.int16(1024)
        A4 = np.float32(440)
        synth = WaveformSynth(
            sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
        )
        plt.figure(figsize=(13, 5))

        waveform1 = 2.0 * synth.gen_single(70, 0.5) + synth.gen_single(
            80, 0
        )
        waveform1 /= waveform1.std()

        plt.subplot(2, 1, 1)
        plt.plot(waveform1, label='singulars')

        waveform2 = synth.gen_multiple(
            np.array(
                [
                    [70, 80],
                    [0.5, 0],
                    [2.0, 1],
                ]
            )
        )
        waveform2 /= waveform2.std()

        plt.plot(waveform2, label='multiples')

        plt.plot(waveform1 - waveform2, label='dif')
        plt.legend()

        waveform1 = (
            10 * synth.gen_single(40, 0.5)
            + 0.5 * synth.gen_single(60, 0)
            + synth.gen_single(80, 0.2)
            + 5 * synth.gen_single(100, 0.6)
            + 3 * synth.gen_single(50, 0.2)
        )
        waveform1 /= waveform1.std()
        plt.subplot(2, 1, 2)
        plt.plot(waveform1, label='singulars')

        waveform2 = synth.gen_multiple(
            np.array(
                [
                    [40, 60, 80, 100, 50],
                    [0.5, 0, 0.2, 0.6, 0.2],
                    [10, 0.5, 1, 5, 3],
                ]
            )
        )
        waveform2 /= waveform2.std()
        plt.plot(waveform2, label='multiples')

        plt.plot(waveform1 - waveform2, label='dif')
        
        plt.legend()
        plt.show()

    @staticmethod
    def train(*args, verbose: bool = False, **kwargs):
        sample_rate = np.float32(16_000)
        buffer_size = np.int16(1024)
        A4 = np.float32(440)

        dataset = SineWaveformDataset(
            1000,
            100,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
        )

        if verbose:
            plt.figure()
            for idx in range(16 * 600, 16 * 600 + 6):
                waveform, pitch = dataset[idx]
                plt.subplot(3, 2, idx + 1 - 16 * 600)
                plt.plot(waveform.detach().cpu().numpy(), label=f"{pitch:.2f}")
                plt.legend()

        trainer = Trainer(MonophonicModel(buffer_size))

        history = trainer.train(1000, dataset)
        # plt.figure()
        # plt.title('history')
        # plt.plot(history)
        plt.show()
        trainer.save_model("monophonic_model.pth")

    @staticmethod
    def defaults(*args, **kwargs):
        print(f"Default device: {torch.get_default_device()}")


def main():
    torch.set_default_device("mps")

    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.autolayout"] = True

    parser = ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "validate", "defaults", "test_generation"],
        help="Mode to run the application in.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to load the model from (required for validate mode).",
    )
    args = parser.parse_args()

    Modes.__dict__[args.mode](verbose=args.verbose)


if __name__ == "__main__":
    main()
