import pytest

import numpy as np
import torch

from rezonance.logger import logger
from rezonance import transforms
from rezonance.defaults import BUFFER_SIZE, SAMPLE_RATE, A4
from rezonance.audioutils.waveform_generators import Instrument
from rezonance.datasets.synth_dataset import InstrumentSynthDataset
from rezonance.objectives import NoteClassifierObjective


def test_classifier():

    dataset = InstrumentSynthDataset(
        0.5,
        5,
        transform=transforms.none(),
        instrument=Instrument.random(
            1.5,
            buffer_size=BUFFER_SIZE,
            sample_rate=SAMPLE_RATE,
            A4=A4,
        ),
        sample_rate=SAMPLE_RATE,
        A4=A4,
        min_pitch=60,
        max_pitch=70,
    )

    objective = NoteClassifierObjective(60, 70, 0.5)

    references = np.arange(60, 70, 0.5)

    references = references.repeat(5, axis=0)

    logger.info("testing dataset size")
    assert len(dataset) == references.shape[0]

    test_classification = torch.zeros((1, 20))

    test_classification[:, 0] = 1
    test_classification[:, 14] = 0

    for reference, (_, y) in zip(references, dataset):  # type: ignore
        assert reference == y

        labels = objective.forward(y)
        loss, _ = objective.loss(test_classification, labels)

        logger.debug(f"{torch.argmax(test_classification)=}")

        logger.debug(f"my loss {loss.item()}")
