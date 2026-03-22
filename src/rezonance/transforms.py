from typing import Callable
import random
import torch
from torch import Tensor
from torch.types import Number

from rezonance.logger import logger
from rezonance.noise_generators import NoiseSynth

Transform = Callable[[Tensor], Tensor]


class OutputTransform:
    def __init__(self):
        pass

    def size(self) -> int:
        """
        The size of the corresponding output
        """
        raise NotImplementedError

    def forward(self, pitch: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, output: Tensor) -> Tensor:
        raise NotImplementedError

class CyclicPitchTransform(OutputTransform):
    def __init__(self, with_octave: bool):
        self.with_octave = with_octave

    def size(self) -> int:
        return 3 if self.with_octave else 2

    def forward(self, pitch: Tensor) -> Tensor:
        return torch.cat(
            [
                torch.sin(pitch * torch.pi / 6),
                torch.cos(pitch * torch.pi / 6),
                pitch / 12 - 1
            ],
            dim=-1
        ) if self.with_octave else torch.cat(
            [
                torch.sin(pitch * torch.pi / 6),
                torch.cos(pitch * torch.pi / 6),
            ],
            dim=-1
        )

    def backward(self, output: Tensor) -> Tensor:
        raise NotImplementedError

# TODO: update documentation
class NoteClassifier(OutputTransform):
    """
    Transforms a pitch value into a tensore of classes,
    where each index represents the pitch, and a value of 1 if present, 0 if not.
    Each octave usually contains 12 notes, (from C -> B), we can increase the resolution
    of our classification by choosing a number for bins_per_octave.
    A value of 10 for `bins_per_pitch` would correspond to classifying notes with resolution
    of 10 cents. (in the equal temperment scale,
    the logarithmic distance between two consecutive notes is 100 cents)

    Parameters:
        min_pitch (Number): corresponds to index 0
        max_pitch (Number): corresponds to last index (included)
        bins_per_octave (int): how many divisions between pitches 12 apart
    """
    def __init__(
        self,
        min_pitch: Number,
        max_pitch: Number,
        pitch_step: Number,
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.bins_per_pitch = 1/pitch_step

    def size(self) -> int:
        return int(
            round(
                (self.max_pitch - self.min_pitch) * self.bins_per_pitch 
            )
        )

    def _get_pitch_index(self, pitch: Tensor) -> Tensor:
        return (
            (pitch - self.min_pitch) * self.bins_per_pitch
        ).round().int()
        

    def forward(self, pitch: Tensor) -> Tensor:
        bins = torch.zeros((pitch.size(0), self.size(),))
        lines = torch.arange(pitch.size(0))
        bins[lines, self._get_pitch_index(pitch)] =  1
        return bins
    
    def backward(self, output: Tensor) -> Tensor:
        raise NotImplementedError


def noise(noise: NoiseSynth) -> Transform:

    def transform_signal(input: Tensor) -> Tensor:
        input = input + noise.generate(input.size(0))
        return input / input.std()

    return transform_signal


def mask(size: int, mask_value: Number) -> Transform:

    def transform_signal(input: Tensor) -> Tensor:
        input = input.clone()

        i = torch.randint(0, input.size(0) - size, (1,))
        j = i + size

        input[i:j] = mask_value
        return input / input.std()

    return transform_signal


def scaling(
    low: Number,
    high: Number,
    buffer_size: int,
) -> Transform:

    indice_multipliers = (high - low) * torch.linspace(
        0, 1, buffer_size
    ) + low

    return lambda input: indice_multipliers * input


def compose(
    *transforms: Transform,
) -> Transform:

    def transform_signal(input: Tensor) -> Tensor:
        for transform in transforms:
            input = transform(input)
        return input

    return transform_signal


def random_choice(
    *transforms: Transform,
) -> Transform:
    def transform_signal(input: Tensor) -> Tensor:
        choice = torch.randint(len(transforms), ())
        return transforms[choice](input)

    return transform_signal


def none():
    """
    A transfrom that does nothing.
    Useful, because you can pass it when you don't want to
    perform a transform instead of None,
    makes the caller simpler by removing the need to check for None
    """
    return lambda x: x
