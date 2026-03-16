import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from rezonance.logger import logger
from rezonance.train_dataset import InstrumentSynthDataset
from rezonance.real_dataset import NSynthDataset
from rezonance.training import Trainer
from rezonance.models.fclinearmodel import FCLinearModel


def run(*args, verbose: bool = False, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0

    train_dataset = InstrumentSynthDataset(
        500,
        1000,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        A4=A4,
    )

    validation_dataset = NSynthDataset(
        Path("data", "nsynth-valid"),
        sample_rate,
        buffer_size,
        element_per_file=5
    )

    
    model = FCLinearModel(buffer_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, criterion, optimizer)

    logger.info("starting training")
    stamp = time.perf_counter()
    trainer.train(train_dataset) #, validation_dataset)
    logger.info(f"finished training {time.perf_counter() - stamp}")

    plt.figure()
    plt.title('history')
    plt.plot(trainer.train_history, label="Training Loss")
    plt.plot(trainer.validation_history, label="Validation Loss")
    plt.legend()
    plt.show()
