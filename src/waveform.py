import numpy as np
import torch
from torch.types import Tensor, Number
from typing import Callable

from src.utils import freq_from_pitch


class NoiseSynth:
    def __init__(
        self,
        power: Number,
        filter: Callable[[Tensor], Tensor],
    ):
        self.power = power
        self.filter = filter

    def __call__(self, buffer_size: int) -> Tensor:
        noise = self.generate_noise(buffer_size)
        noise_fft = torch.fft.rfft(noise)
        filtered_freq = self.filter(torch.fft.rfftfreq(buffer_size))
        filtered_noise_fft = noise_fft * filtered_freq
        noise = torch.fft.irfft(filtered_noise_fft)
        return self.power * noise / noise.std()

    @classmethod
    def generate_noise(cls, buffer_size: int) -> Tensor:
        return torch.normal(
            mean=torch.zeros(buffer_size),
            std=1.0,
        )


class Noise:
    @classmethod
    def brown(cls, power: Number) -> NoiseSynth:
        return NoiseSynth(
            power=power,
            filter=lambda freq: 1
            / torch.where(freq == 0, float("inf"), freq),
        )

    @classmethod
    def pink(cls, power: Number) -> NoiseSynth:
        return NoiseSynth(
            power=power,
            filter=lambda freq: 1
            / torch.where(freq == 0, float("inf"), torch.sqrt(freq)),
        )

    class white(NoiseSynth):
        def __init__(self, power: Number):
            self.power = power

        def __call__(self, buffer_size: int) -> Tensor:
            return self.generate_noise(buffer_size)

    @classmethod
    def blue(cls, power: Number) -> NoiseSynth:
        return NoiseSynth(
            power=power, filter=lambda freq: torch.sqrt(freq)
        )

    @classmethod
    def violet(cls, power: Number) -> NoiseSynth:
        return NoiseSynth(power=power, filter=lambda freq: freq)


class WaveformSynth:
    """
    A simple waveform synthesizer generating sinewaves from pitch and phase.
    Parameters:
        sample_rate (float): the sample rate of the generated waveform
        buffer_size (int): the buffer size of the generated waveform
        A4 (float): the reference frequency of the A4 note
    """

    def __init__(
        self,
        *,
        sample_rate: Number,
        buffer_size: int,
        A4: Number,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.A4 = A4

    def gen_single(
        self,
        pitch: Number,
        phase: Number,
    ) -> Tensor:
        """
        Generate sinusoidal waveform
        Parameters:
            pitch (float): the pitch number (logarithmic scale) 69 represents A4
            phase (float): the phase `[-1, 1]`
        Returns:
            Sine wave, not normalized, consider scaling with std
        """
        frequency = freq_from_pitch(pitch, A4=self.A4)  # type: ignore
        linspace = torch.linspace(
            0,
            self.buffer_size / self.sample_rate,
            self.buffer_size,
        )
        return torch.sin((frequency * linspace + phase) * np.pi)

    def gen_multiple(
        self,
        params: Tensor,
    ) -> Tensor:
        """
        Generating waveform as sum of sinewaves from a distriution of parameters.
        Parameters:
            params (matrix): representing sinewaves params, size `3 * n`
                n being the number of sines.
                - The first row is the pitch.
                - The second row is the phase.
                - The third row is the power.
        Returns:
            The sum of all sinewaves, the result is not scaled or normalized, consider dividing by std
        """
        linspace = torch.linspace(
            0,
            self.buffer_size / self.sample_rate,
            self.buffer_size,
        )

        return (
            params[2, None]  # power
            * np.sin(
                (
                    linspace[:, None]
                    @ freq_from_pitch(
                        params[0, None],  # pitch
                        A4=self.A4,
                    )  # frequency
                    + params[1, None]  # phase
                )
                * np.pi
            )
        ).sum(axis=1)  # summing all sines together
