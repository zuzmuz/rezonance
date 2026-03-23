from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rezonance.logger import logger
from rezonance.defaults import BUFFER_SIZE, SAMPLE_RATE
from rezonance.datasets.real_dataset import H5Dataset, FileDataset


def main():

    h5_dataset = H5Dataset(Path("data", "test_dataset.h5"))

    nsynth_dataset = FileDataset(
        Path("data", "nsynth-test"),
        SAMPLE_RATE,
        BUFFER_SIZE,
        element_per_file=2,
    )

    logger.info(
        f"Sizes of datasets: h5={len(h5_dataset)}, nsynth={len(nsynth_dataset)}"
    )

    rand_indices = np.random.randint(0, len(h5_dataset), (3,))

    plt.figure()

    for i, index in enumerate(rand_indices):
        logger.info(f"Running test on item {index} of the dataset")

        h5_signal, h5_pitch = h5_dataset[index]
        ns_signal, ns_pitch = nsynth_dataset[index]

        plt.subplot(rand_indices.shape[0], 2, (i * 2) + 1)
        plt.plot(h5_signal)
        plt.title(f"Pitch {h5_pitch}")

        plt.subplot(rand_indices.shape[0], 2, (i * 2) + 2)
        plt.plot(ns_signal)
        plt.title(f"Pitch {ns_pitch}")

    plt.show()


if __name__ == "__main__":
    main()
