from argparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
import sounddevice as sd
from torch.utils.data import Dataset, DataLoader

from src.generator import Synthesizer
from src.music import Melody, Note
from src.training import WaveformDataset, MonophonicModel, Trainer


class Modes:

    @staticmethod
    def train():

        sample_rate = np.float32(16_000)
        buffer_size = np.int16(1024)
        A4 = np.float32(440)

        dataset = WaveformDataset(
            1000,
            4,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
            seed=42
        )

        trainer = Trainer(MonophonicModel(buffer_size))

        trainer.train(1000, dataset)
        trainer.save_model("monophonic_model.pth")
        

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "validate"],
        help="Mode to run the application in.",
    )
    args = parser.parse_args()

    Modes.__dict__[args.mode]()


if __name__ == "__main__":
    main()
