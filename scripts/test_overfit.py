import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from rezonance.defaults import SAMPLE_RATE, BUFFER_SIZE, A4
from rezonance.utils import current_device
from rezonance import transforms
from rezonance.waveform import Instrument
from rezonance.train_dataset import InstrumentSynthDataset
from rezonance.models.convmodel import ConvModel 
from rezonance.models.smalltestmodel import SmallTestModel
from rezonance.training import Trainer


def main():

    torch.set_default_device(current_device)

    min_pitch = 36
    max_pitch = 84  # not included

    dataset = InstrumentSynthDataset(
        1,
        2000,
        transform=transforms.none(),
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
    )

    output_transform = transforms.NoteClassifier(
        min_pitch, max_pitch, 1
    )
    
    # output_transform = transforms.CyclicPitchTransform(False)

    model = ConvModel(output_transform.size())

    # model = SmallTestModel(BUFFER_SIZE, output_transform.size())

    criterion = output_transform.criterion()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        output_transform,
    )

    trainer.overfit_test(dataset)


if __name__ == "__main__":
    main()
