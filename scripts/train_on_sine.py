import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt

from src.logger import logger
from src.train_dataset import InstrumentSynthDataset
from src.real_dataset import NSynthDataset
from src.training import Trainer
from src.models.fclinearmodel import FCLinearModel
from src.waveform import Instrument


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    train_dataset = ConcatDataset(
        [
            InstrumentSynthDataset(
                500,
                2000,
                instrument=Instrument.random(
                    1.5,
                    buffer_size=buffer_size,
                    sample_rate=sample_rate,
                    A4=A4,
                ),
                sample_rate=sample_rate,
                A4=A4,
            ),
            InstrumentSynthDataset(
                200,
                500,
                instrument=Instrument.random(
                    2,
                    buffer_size=buffer_size,
                    sample_rate=sample_rate,
                    A4=A4,
                ),
                sample_rate=sample_rate,
                A4=A4,
            ),
            InstrumentSynthDataset(
                100,
                100,
                instrument=Instrument.random(
                    1,
                    buffer_size=buffer_size,
                    sample_rate=sample_rate,
                    A4=A4,
                ),
                sample_rate=sample_rate,
                A4=A4,
            ),
        ]
    )

    validation_dataset = NSynthDataset(
        Path("data", "nsynth-valid"),
        sample_rate,
        buffer_size,
        element_per_file=5,
    )

    model = FCLinearModel(buffer_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, criterion, optimizer)

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
