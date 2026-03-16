from typing import Callable
import torch
from torch import Tensor
from torch.types import Number

from rezonance.noise_generators import NoiseSynth


class Augmentation:
    def __init__(
        self, augmentation: Callable[[Tensor], Tensor], chance: Number
    ):
        self.augmentation = augmentation
        self.chance = chance

    def __call__(self, input: Tensor) -> Tensor:
        """
        Applies augmentation by chance
        """
        if torch.rand(1) < self.chance:
            return self.augmentation(input)
        return input

    @classmethod
    def noise(cls, noise: NoiseSynth, chance: Number) -> Augmentation:
        def noise_signal(input: Tensor) -> Tensor:
            input += noise.generate(input.size(0))
            return input / input.std()

        return Augmentation(noise_signal, chance)

    @classmethod
    def mask(
        cls, size: int, mask_value: Number, chance: Number
    ) -> Augmentation:
        def mask_signal(input: Tensor) -> Tensor:
            i = torch.randint(0, input.size(0) - size, (1,))
            j = i + size

            input[i:j] = mask_value
            return input / input.std()

        return Augmentation(mask_signal, chance)

    @classmethod
    def scaling(
        cls,
        low: Number,
        high: Number,
        buffer_size: int,
        chance: Number,
    ) -> Augmentation:
        indice_multipliers = (high - low) * torch.linspace(
            0, 1, buffer_size
        ) + low

        return Augmentation(
            augmentation=lambda input: indice_multipliers * input,
            chance=chance,
        )
