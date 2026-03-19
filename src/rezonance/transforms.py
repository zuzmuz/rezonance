from typing import Callable
import random
import torch
from torch import Tensor
from torch.types import Number

from rezonance.noise_generators import NoiseSynth

Transform = Callable[[Tensor], Tensor]


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
