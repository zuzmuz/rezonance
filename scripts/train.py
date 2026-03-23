from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset

from rezonance.models.smalltestmodel import SmallTestModel
from rezonance.utils import current_device
from rezonance.logger import logger
from rezonance.defaults import SAMPLE_RATE, BUFFER_SIZE, A4
from rezonance.models.convmodel import ConvModel
from rezonance.audioutils.waveform_generators import Instrument
from rezonance.audioutils.noise_generators import Noise
from rezonance.datasets.synth_dataset import InstrumentSynthDataset
from rezonance.datasets.real_dataset import H5Dataset
from rezonance.models.fcmodel import FCModel
from rezonance.training import Trainer
from rezonance.objectives import (
    ClassificationMetric,
    NoteClassifierObjective,
)
from rezonance import transforms


def main():

    torch.set_default_device(current_device)

    # torch.manual_seed(5)

    logger.info(f"Using device: {torch.get_default_device()}")

    logger.info("Generating train synthetic dataset")

    multiplier = 1

    min_pitch = 36
    max_pitch = 84  # not included

    train_dataset = ConcatDataset(
        [
            InstrumentSynthDataset(
                1 / 4,
                multiplier * 250,
                transform=transforms.random_choice(
                    transforms.none(),
                    transforms.noise(
                        Noise.brown(0.1) + Noise.violet(0.02)
                    ),
                    transforms.compose(
                        transforms.noise(
                            Noise.brown(0.1) + Noise.violet(0.02)
                        ),
                        transforms.scaling(1, 0.8, BUFFER_SIZE),
                    ),
                    transforms.noise(Noise.white(0.05)),
                    transforms.scaling(1, 0.5, BUFFER_SIZE),
                    transforms.mask(50, 0),
                ),
                instrument=Instrument.random(
                    1.5,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
                min_pitch=min_pitch,
                max_pitch=max_pitch,
            ),
            InstrumentSynthDataset(
                1 / 2,
                multiplier * 125,
                transform=transforms.random_choice(
                    transforms.noise(Noise.brown(0.1)),
                    transforms.scaling(1, 0.7, BUFFER_SIZE),
                ),
                instrument=Instrument.random(
                    2,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
                min_pitch=min_pitch,
                max_pitch=max_pitch,
            ),
            InstrumentSynthDataset(
                1,
                multiplier * 75,
                transform=transforms.none(),
                instrument=Instrument.random(
                    1,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
                min_pitch=min_pitch,
                max_pitch=max_pitch,
            ),
        ]
    )

    logger.info("Creating real validation dataset")

    validation_dataset = H5Dataset(
        Path(
            "data",
            f"valid_dataset_filtered_{min_pitch}_{max_pitch}.h5",
        )
    )

    objective = NoteClassifierObjective(min_pitch, max_pitch, 1 / 4)
    # model = FCLinearModel(BUFFER_SIZE, objective.output_size())
    model = SmallTestModel(BUFFER_SIZE, objective.output_size())

    # model = ConvModel(objective.output_size())

    # the output transform chooses its loss function

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, optimizer, objective)

    logger.info("starting training")

    validate_every = 1
    try:
        trainer.train(
            train_dataset,
            validation_dataset,
            log_epochs=1,
            log_batch=50,
            validate_every=validate_every,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted — saving current model state...")
    finally:
        model_path = Path("saved_models", "model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved to {model_path}")

    # TODO: fix plotting for specific objective
    plt.figure()
    plt.title("Loss History")
    plt.plot(
        [metric.loss for metric in trainer.train_history],
        label="Training Loss",
    )
    plt.plot(
        np.arange(
            0,
            validate_every * len(trainer.validation_history),
            validate_every,
        ),
        [metric.loss for metric in trainer.validation_history],
        label="Validation Loss",
    )
    plt.legend()
    plt.savefig(Path("figures", "loss.png"))

    if isinstance(objective.get_metric(), ClassificationMetric):
        plt.figure()
        plt.title("Accuracy History")
        plt.plot(
            [metric.accuracy for metric in trainer.train_history],
            label="Training Accuracy",
        )
        plt.plot(
            np.arange(
                0,
                validate_every * len(trainer.validation_history),
                validate_every,
            ),
            [metric.accuracy for metric in trainer.validation_history],
            label="Validation Accuracy",
        )
        plt.savefig(Path("figures", "accuracy.png"))



if __name__ == "__main__":
    main()
