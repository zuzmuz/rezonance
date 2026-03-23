"""
Module to generate synthetic datasets
"""

import torch
from torch.types import Number
from torch.utils.data import Dataset

from rezonance.utils import current_device
from rezonance.logger import logger
from rezonance.audioutils.pitch_utils import (
    pitch_from_freq,
)
from rezonance.audioutils.waveform_generators import InstrumentSynth
from rezonance.transforms import Transform


class InstrumentSynthDataset(Dataset):
    """
    Synthetic dataset generator

    Parameters:
        pitch_step (Number): the pitch resolution, a value of 1 corresponds to half note.
        per_pitch (int): the number of signals to generate per pitch
    KeywordArguments:
        transform (Transform): transform to run on generated data
        instrument (InstrumentSynth): the instrument to use for data generation
        sample_rate (Number): the sampling frequency
        A4 (Number): the A4 reference
        min_pitch (Number): the minimum pitch to generate
        max_pitch (Number): the maximum pitch to generate
        seed (int): random seed
    """
    def __init__(
        self,
        pitch_step: Number,
        per_pitch: int,
        *,
        transform: Transform,
        instrument: InstrumentSynth,
        sample_rate: Number = 16_000,
        A4: Number = 440,
        min_pitch: Number = 20,
        max_pitch: Number | None = None,
        seed: int | None = None,
    ):

        if seed:
            torch.manual_seed(seed)

        self.transform = transform
        self.instrument = instrument

        max_possible_pitch = pitch_from_freq(
            0.25 * sample_rate, A4=A4
        )

        if not max_pitch:
            max_pitch = max_possible_pitch
        elif max_pitch > max_possible_pitch:
            logger.warning(
                "Provided max_pitch %.2f "
                "exceeds the maximum possible pitch %.2f"
                "for the given sample rate %.2f.",
                max_pitch,
                max_possible_pitch,
                sample_rate,
            )

        logger.debug(f"Generating pitches")
        self.pitches = torch.arange(
            min_pitch, max_pitch, pitch_step, dtype=torch.float32
        )

        logger.debug(
            f"Generated pitches with size {self.pitches.size(0)}"
        )

        self.nb_harm_dist = per_pitch

        logger.debug("Generating signals")
        self.data = self.instrument.generate(
            self.pitches,
            per_pitch=per_pitch,
        )
        logger.debug(
            f"Generated signals with size = {self.data.size(0)}"
        )

        self.data /= self.data.std(dim=1, keepdim=True)

        logger.debug("Moving tensor to device")
        self.data.to(current_device)

        logger.debug("Done generating data")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        pitch = self.pitches[idx // self.nb_harm_dist]

        waveform = self.transform(self.data[idx])
        # waveform = self.data[idx]

        return waveform, pitch.unsqueeze(0)
