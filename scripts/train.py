from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset

from rezonance.utils import current_device
from rezonance.logger import logger
from rezonance.defaults import SAMPLE_RATE, BUFFER_SIZE, A4
from rezonance.models.convmodel import ConvModel
from rezonance.waveform import Instrument
from rezonance.train_dataset import InstrumentSynthDataset
# from rezonance.real_dataset import NSynthDataset
from rezonance.models.fclinearmodel import FCLinearModel
from rezonance.noise_generators import Noise
from rezonance.training import Trainer
from rezonance import transforms


def main():

    torch.set_default_device(current_device)
    logger.info(f"Using device: {torch.get_default_device()}")

    logger.info("Generating train synthetic dataset")
    
    train_dataset = ConcatDataset(
        [
            InstrumentSynthDataset(
                190,
                1000,
                transform=transforms.random_choice(
                    transforms.none(),
                    transforms.noise(Noise.brown(0.1) + Noise.violet(0.02)),
                    transforms.compose(
                        transforms.noise(Noise.brown(0.1) + Noise.violet(0.02)),
                        transforms.scaling(1, 0.8, BUFFER_SIZE)
                    ),
                    transforms.noise(Noise.white(0.05)),
                    transforms.scaling(1, 0.5, BUFFER_SIZE),
                    transforms.mask(50, 0)
                ),
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
                150,
                500,
                transform=transforms.random_choice(
                    transforms.none(),
                    transforms.noise(Noise.brown(0.1)),
                    transforms.scaling(1, 0.7, BUFFER_SIZE)
                ),
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
                110,
                300,
                transform=transforms.none(),
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

    # validation_dataset = NSynthDataset(
    #     Path("data", "nsynth-valid"),
    #     SAMPLE_RATE,
    #     BUFFER_SIZE,
    #     element_per_file=5,
    # )

    # model = FCLinearModel(BUFFER_SIZE)
    model = ConvModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
    )

    logger.info("starting training")
    trainer.train(
        train_dataset,
        None, # validation_dataset,
        log_epochs=1,
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


if __name__ == "__main__":
    main()
