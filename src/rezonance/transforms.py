"""
Module that contains input transform fumctions for data augmentation.
"""


from typing import Callable
import torch
from torch import Tensor
from torch.types import Number

from rezonance.logger import logger
from rezonance.noise_generators import NoiseSynth

Transform = Callable[[Tensor], Tensor]
"""
Callables on input tensors for data augmentation
"""


def noise(noise: NoiseSynth) -> Transform:
    """
    Create data augmentation transform that adds noise to input signal
    """
    def transform_signal(input: Tensor) -> Tensor:
        input = input + noise.generate(input.size(0))
        return input / input.std()

    return transform_signal


def mask(size: int, mask_value: Number) -> Transform:
    """
    Create data augmentation transform that mask out random time band in signal
    """
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
    """
    Create a gain envelop transform
    """
    indice_multipliers = (high - low) * torch.linspace(
        0, 1, buffer_size
    ) + low

    return lambda input: indice_multipliers * input


def compose(
    *transforms: Transform,
) -> Transform:
    """
    Create a sequence of transforms
    """
    def transform_signal(input: Tensor) -> Tensor:
        for transform in transforms:
            input = transform(input)
        return input

    return transform_signal


def random_choice(
    *transforms: Transform,
) -> Transform:
    """
    Create a transform that represent a uniformely distributed
    choice between multiple transforms
    """
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
