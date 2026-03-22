from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from rezonance.logger import logger
from rezonance.defaults import BUFFER_SIZE, SAMPLE_RATE
from rezonance.real_dataset import NSynthDataset, H5Dataset


def store(
    data_set: Dataset,
    path: Path,
    *,
    min_pitch: Number = 20,
    max_pitch: Number = 120
):
    tensors = []
    labels = []

    for signal, pitch in data_set:
        if min_pitch <= pitch <= max_pitch:
            tensors.append(signal.to("cpu").numpy())
            labels.append(pitch.to("cpu").numpy())

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.stack(tensors))  # (N, C, T)
        f.create_dataset("labels", data=np.array(labels))


def main():

    # valid_dataset = NSynthDataset(
    #     Path("data", "nsynth-valid"),
    #     SAMPLE_RATE,
    #     BUFFER_SIZE,
    #     element_per_file=2
    # )
    #
    #
    # test_dataset = NSynthDataset(
    #     Path("data", "nsynth-test"),
    #     SAMPLE_RATE,
    #     BUFFER_SIZE,
    #     element_per_file=2
    # )

    valid_dataset = H5Dataset(
        Path("data", "valid_dataset.h5"),
    )

    test_dataset = H5Dataset(
        Path("data", "test_dataset.h5"),
    )

    logger.info("storing validation dataset")
    store(
        valid_dataset,
        Path("data", "valid_dataset_filtered_36_83.h5"),
        min_pitch=36,
        max_pitch=83
    )
    logger.info("storing test dataset")
    store(
        test_dataset,
        Path("data", "test_dataset_filtered_36_83.h5"),
        min_pitch=36,
        max_pitch=83
    )


if __name__ == '__main__':
    main()
