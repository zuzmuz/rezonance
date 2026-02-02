from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

from src.training import SineWaveformDataset, MonophonicModel, Trainer


class Modes:
    @staticmethod
    def train(verbose: bool = False):
        sample_rate = np.float32(16_000)
        buffer_size = np.int16(1024)
        A4 = np.float32(440)

        dataset = SineWaveformDataset(
            1000,
            4,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
        )

        if verbose:
            plt.figure()
            for idx in range(16*600, 16*600 + 6):
                waveform, pitch = dataset[idx]
                plt.subplot(3, 2, idx+1 - 16*600)
                plt.plot(waveform.numpy(), label=f'{pitch:.2f}')
                plt.legend()

        trainer = Trainer(MonophonicModel(buffer_size))

        # history = trainer.train(1, dataset)
        # plt.figure()
        # plt.title('history')
        # plt.plot(history)
        plt.show()
        # trainer.save_model("monophonic_model.pth")


def main():

    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.autolayout'] = True

    parser = ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "validate"],
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
