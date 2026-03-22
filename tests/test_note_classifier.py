import pytest

import numpy as np
import torch

from rezonance.logger import logger
from rezonance import transforms
from rezonance.defaults import BUFFER_SIZE, SAMPLE_RATE, A4
from rezonance.waveform import Instrument
from rezonance.train_dataset import InstrumentSynthDataset


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

    output_transform = transforms.NoteClassifier(60, 70, 0.5)

    references = np.arange(60, 70, 0.5)

    references = references.repeat(5, axis=0)
    
    logger.info("testing dataset size")
    assert len(dataset) == references.shape[0]

    criterion = output_transform.criterion()
    
    test_classification = torch.zeros((1, 20))

    test_classification[:, 0] = 1
    test_classification[:, 14] = 0
    
    for (reference, (_, y)) in zip(references, dataset): # type: ignore
        assert reference == y
        

        transformed_output = output_transform.forward(y)
        # logger.debug(f"{test_classification=}, {transformed_output=}")
        loss = criterion(test_classification, transformed_output)
        
        logger.debug(f"{torch.argmax(test_classification)=}")

        logger.debug(f"my loss {loss.item()}")
        # break
        # assert transformed_output.size(0) == 20


