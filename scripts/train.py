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
from rezonance.waveform import Instrument
from rezonance.train_dataset import InstrumentSynthDataset
from rezonance.real_dataset import H5Dataset, NSynthDataset
from rezonance.models.fclinearmodel import FCLinearModel
from rezonance.noise_generators import Noise
from rezonance.training import Trainer
from rezonance import transforms


def main():

    torch.set_default_device(current_device)

    torch.manual_seed(5)

    logger.info(f"Using device: {torch.get_default_device()}")

    logger.info("Generating train synthetic dataset")
    
    multiplier = 1
    train_dataset = ConcatDataset(
        [
            InstrumentSynthDataset(
                190,
                multiplier*500,
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
                max_pitch=100,
            ),
            InstrumentSynthDataset(
                150,
                multiplier*250,
                transform=transforms.random_choice(
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
                max_pitch=100,
            ),
            InstrumentSynthDataset(
                110,
                multiplier*150,
                transform=transforms.none(),
                instrument=Instrument.random(
                    1,
                    buffer_size=BUFFER_SIZE,
                    sample_rate=SAMPLE_RATE,
                    A4=A4,
                ),
                sample_rate=SAMPLE_RATE,
                A4=A4,
                max_pitch=100,
            ),
        ]
    )

    logger.info("Creating real validation dataset")

    validation_dataset = H5Dataset(
        Path("data", "valid_dataset.h5")
    )
    
    output_transform = transforms.NoteClassifier(20, 100, 120)
    # model = FCLinearModel(BUFFER_SIZE, 2)
    # model = SmallTestModel(BUFFER_SIZE)
    model = ConvModel(output_transform.size())
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        output_transform
    )

    logger.info("starting training")

    validate_every = 2
    try:
        trainer.train(
            train_dataset,
            validation_dataset,
            log_epochs=1,
            log_batch=20,
            validate_every=validate_every,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted — saving current model state...")
    finally:
        model_path = Path("saved_models", "model.pth")
        torch.save(
            model.state_dict(),
            model_path
        )
        logger.info(f"Saved to {model_path}")
    

    plt.figure()
    plt.title("history")
    plt.plot(trainer.train_history, label="Training Loss")
    plt.plot(
        np.arange(
            0,
            validate_every * len(trainer.validation_history),
            validate_every
        ),
        trainer.validation_history,
        label="Validation Loss",
    )
    plt.legend()
    plt.show()
    # plt.savefig(Path("figures", "loss.png"))


if __name__ == "__main__":
    main()
