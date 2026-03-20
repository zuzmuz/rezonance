import torch
from torch.types import Number
from torch import Tensor
from torch.utils.data import Dataset

from rezonance.logger import logger
from rezonance.utils import (
    pitch_from_freq,
    current_device,
)
from rezonance.waveform import InstrumentSynth
from rezonance.transforms import Transform


class InstrumentSynthDataset(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_harm_dist: int,
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
        self.pitches = torch.linspace(
            min_pitch, max_pitch, nb_pitches, dtype=torch.float32
        )

        logger.debug(f"Generated pitches with size {self.pitches.size(0)}")

        self.nb_harm_dist = nb_harm_dist

        logger.debug("Generating signals")
        self.data = self.instrument.generate(
            self.pitches,
            per_pitch=nb_harm_dist,
        )
        logger.debug(f"Generated signals with size = {self.data.size(0)}")

        self.data /= self.data.std(dim=1, keepdim=True)
        
        logger.debug("Moving tensor to device")
        self.data.to(current_device)

        logger.debug("Done generating data")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        pitch = self.pitches[idx // self.nb_harm_dist]

        waveform = self.transform(self.data[idx])

        return waveform, pitch
