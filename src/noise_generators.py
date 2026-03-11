import torch
from torch.types import Tensor, Number
from typing import Callable
from src.utils import current_device


class NoiseSynth:
    """
    A callable synth that generates noise based on a filter
    """

    def __init__(
        self,
        power: Number,
        filter: Callable[[Tensor], Tensor],
    ):
        self.power = power
        self.filter = filter

    def __call__(self, buffer_size: int) -> Tensor:
        """
        Generate a noise buffer of given size.
        Parameters:
            buffer_size (int): the size of the buffer to generate
        Returns:
            A noise buffer of size `buffer_size`, the noise profile is determined by the filter
        """
        noise = self.generate_white_noise(buffer_size)
        noise_fft = torch.fft.rfft(noise)
        filtered_freq = self.filter(torch.fft.rfftfreq(buffer_size))
        filtered_noise_fft = noise_fft * filtered_freq
        noise = torch.fft.irfft(filtered_noise_fft)
        return self.power * noise / noise.std()

    @classmethod
    def generate_white_noise(cls, buffer_size: int) -> Tensor:
        return torch.normal(
            mean=torch.zeros(buffer_size),
            std=1.0,
            generator=torch.Generator(device=current_device)
        )

    def __add__(self, other: NoiseSynth) -> NoiseSynth:
        return CompositeNoiseSynth(synths=[self, other])


class CompositeNoiseSynth(NoiseSynth):
    """
    A utility to combine multiple noise synths together
    """

    def __init__(self, synths: list[NoiseSynth]):
        self.synths = synths

    def __call__(self, buffer_size: int) -> Tensor:
        return torch.stack(
            [synth(buffer_size) for synth in self.synths]
        ).sum(dim=0)

    def __add__(self, other: NoiseSynth) -> NoiseSynth:
        self.synths.append(other)
        return self


class Noise:
    @classmethod
    def brown(cls, power: Number) -> NoiseSynth:
        """
        Returns a brown noise

        The filter shape of the brown noise takes the form of 1/freq
        Parameters:
            power (Number): the amplitude of the noise
        Returns:
            A noise synth that will generate a brown noise when called

        """
        return NoiseSynth(
            power=power,
            filter=lambda freq: 1 / torch.where(freq == 0, 1, freq),
        )

    @classmethod
    def pink(cls, power: Number) -> NoiseSynth:
        """
        Returns a pink (low frequency) noise

        The filter shape of the brown noise takes the form of 1/√freq
        Parameters:
            power (Number): the amplitude of the noise
        Returns:
            A noise synth that will generate a pink noise when called

        """
        return NoiseSynth(
            power=power,
            filter=lambda freq: (
                1 / torch.where(freq == 0, 1, torch.sqrt(freq))
            ),
        )

    class white(NoiseSynth):
        """
        Returns a white noise

        The filter shape of the brown noise takes the form of 1
        Parameters:
            power (Number): the amplitude of the noise
        Returns:
            A noise synth that will generate a white noise when called

        """

        def __init__(self, power: Number):
            self.power = power

        def __call__(self, buffer_size: int) -> Tensor:
            return self.power * self.generate_white_noise(buffer_size)

    @classmethod
    def blue(cls, power: Number) -> NoiseSynth:
        """
        Returns a blue (high frequency) noise

        The filter shape of the brown noise takes the form of √freq
        Parameters:
            power (Number): the amplitude of the noise
        Returns:
            A noise synth that will generate a blue noise when called

        """
        return NoiseSynth(
            power=power, filter=lambda freq: torch.sqrt(freq)
        )

    @classmethod
    def violet(cls, power: Number) -> NoiseSynth:
        """
        Returns a violet (high frequency) noise

        The filter shape of the brown noise takes the form of freq
        Parameters:
            power (Number): the amplitude of the noise
        Returns:
            A noise synth that will generate a violet noise when called

        """
        return NoiseSynth(power=power, filter=lambda freq: freq)
