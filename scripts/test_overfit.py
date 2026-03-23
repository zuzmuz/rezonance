import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from rezonance.defaults import SAMPLE_RATE, BUFFER_SIZE, A4
from rezonance.utils import current_device
from rezonance import transforms
from rezonance.audioutils.waveform_generators import Instrument
from rezonance.datasets.synth_dataset import InstrumentSynthDataset
from rezonance.models.convmodel import ConvModel 
from rezonance.models.smalltestmodel import SmallTestModel
from rezonance.training import Trainer
from rezonance.objectives import NoteClassifierObjective, CyclicPitchObjective

def main():

    torch.set_default_device(current_device)

    min_pitch = 36
    max_pitch = 72  # not included

    dataset = InstrumentSynthDataset(
        1,
        100,
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

    objective = NoteClassifierObjective(
        min_pitch, max_pitch, 1
    )
    
    # objective = CyclicPitchObjective(False)
    # output_transform = transforms.CyclicPitchTransform(False)

    # model = ConvModel(objective.output_size())

    model = SmallTestModel(BUFFER_SIZE, objective.output_size())

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    trainer = Trainer(
        model,
        optimizer,
        objective,
    )

    trainer.overfit_test(dataset)


if __name__ == "__main__":
    main()
