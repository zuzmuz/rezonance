from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset

from rezonance.logger import logger
from rezonance.defaults import SAMPLE_RATE, BUFFER_SIZE, A4
from rezonance.waveform import Instrument
from rezonance.train_dataset import InstrumentSynthDataset
from rezonance.real_dataset import NSynthDataset
from rezonance.models.fclinearmodel import FCLinearModel
from rezonance.training import Trainer


def main():

    torch.set_default_device("mps")
    logger.info(f"Using device: {torch.get_default_device()}")

    logger.info("Generating train synthetic dataset")

    train_dataset = ConcatDataset(
        [
            InstrumentSynthDataset(
                500,
                2000,
                instrument=Instrument.random(
                    1.5,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
            InstrumentSynthDataset(
                200,
                500,
                instrument=Instrument.random(
                    2,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
            InstrumentSynthDataset(
                100,
                100,
                instrument=Instrument.random(
                    1,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
            ),
        ]
    )

    logger.info("Creating real validation dataset")

    validation_dataset = NSynthDataset(
        Path("data", "nsynth-valid"),
        SAMPLE_RATE,
        BUFFER_SIZE,
        element_per_file=5,
    )

    model = FCLinearModel(BUFFER_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, criterion, optimizer, augmentations=[])

    logger.info("starting training")
    trainer.train(
        train_dataset,
        validation_dataset,
        log_epochs=2,
        validate_every=10,
    )

    plt.figure()
    plt.title("history")
    plt.plot(trainer.train_history, label="Training Loss")
    plt.plot(
        np.arange(0, len(trainer.train_history), 10),
        trainer.validation_history,
        label="Validation Loss",
    )
    plt.legend()
    plt.savefig(Path("figures", "loss.png"))


if __name__ == '__main__':
    main()
