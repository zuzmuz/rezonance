from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rezonance.logger import logger
from rezonance.defaults import BUFFER_SIZE, SAMPLE_RATE
from rezonance.models.convmodel import ConvModel
from rezonance.real_dataset import NSynthDataset


def main():

    test_dataset = NSynthDataset(
        Path("data", "nsynth-test"),
        SAMPLE_RATE,
        BUFFER_SIZE,
        element_per_file=5,
    )

    model = ConvModel()
    model.load_state_dict()

    rand_indices = np.random.randint(0, len(test_dataset), (3,))

    plt.figure()

    for i, index in enumerate(rand_indices):
        logger.info(f"Running test on item {index} of the dataset")

        signal, pitch = test_dataset[index]

        predicted = model(signal)
        plt.subplot(rand_indices.shape[0], 1, i + 1)
        plt.plot(signal)
        plt.title(f"Pitch = {pitch}, predicted = {predicted}")

    plt.show()
