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
        """
        Generate a noise buffer of given size.
        Parameters:
            buffer_size (int): the size of the buffer to generate
        Returns:
            A noise buffer of size `buffer_size`, the noise profile is determined by the filter
        """
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

    def __add__(self, other: NoiseSynth) -> NoiseSynth:
        return CompositeNoiseSynth(synths=[self, other])


class CompositeNoiseSynth(NoiseSynth):
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
        return NoiseSynth(
            power=power,
            filter=lambda freq: 1 / torch.where(freq == 0, 1, freq),
        )

    @classmethod
    def pink(cls, power: Number) -> NoiseSynth:
        return NoiseSynth(
            power=power,
            filter=lambda freq: 1
            / torch.where(freq == 0, 1, torch.sqrt(freq)),
        )

    class white(NoiseSynth):
        def __init__(self, power: Number):
            self.power = power

        def __call__(self, buffer_size: int) -> Tensor:
            return self.power * self.generate_noise(buffer_size)

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

    def gen_mono(
        self,
        params: Tensor,
    ) -> Tensor:
        """
        Generate monophonic sinusoidal waveform,
        Parameters:
            params (tensor): representing sinewave params, size `(n, 2)`
                - n being the number of signals required.
                - 2:
                    - index 0 is pitch
                    - index 1 is phase
        Returns:
            Sine waves size `(n, buffer_size)`,
            not normalized, consider scaling with std
        """

        # Generating the linspace for the waveform, this will represent time
        linspace = torch.linspace(
            0,
            self.buffer_size / self.sample_rate,
            self.buffer_size,
        ).unsqueeze(
            0
        )  # adding a dimension, linspace shape `(1, buffer_size)`

        # Creating frequencies and phases matrices from params, shape(n, 1)
        frequencies = freq_from_pitch(
            params[:, 0].unsqueeze(1),  # pitch
            A4=self.A4,
        )
        phases = params[:, 1].unsqueeze(1)

        return torch.sin((phases + frequencies @ linspace) * np.pi)

    def gen_poly(
        self,
        params: Tensor,
    ) -> Tensor:
        """
        Generating polyphonic waveform as sum of sinewaves from a distriution of parameters.
        Parameters:
            params (tensor): representing sinewaves params, size `(n, p, 3)`
                - n being the number of signals required.
                - p the number of harmonics per signal.
                - 3:
                    - The first row is the pitch.
                    - The second row is the phase.
                    - The third row is the power.
        Returns:
            A tensor containing all signals with the sum of all harmonics,
            of shape `(n, buffer_size)`,
            the result is not scaled or normalized, consider dividing by std
        """

        # Generating the linspace for the waveform, this will represent time
        linspace = (
            torch.linspace(
                0,
                self.buffer_size / self.sample_rate,
                self.buffer_size,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(params.size(0), 1, 1)
        )
        # adding a dimension and repeating for each signal,
        # linspace shape `(n, 1, buffer_size)`
        print(f'{linspace.shape=}')

        # Creating frequencies and phases matrices from params, shape(n, p, 1)
        frequencies = freq_from_pitch(
            params[:, :, 0].unsqueeze(2), A4=self.A4
        )
        phases = params[:, :, 1].unsqueeze(2)
        powers = params[:, :, 2].unsqueeze(2)
        print(f'{frequencies.shape=}, {phases.shape=}, {powers.shape=}')
        
        FT = torch.baddbmm(phases, frequencies, linspace)
        print(f'{FT.shape}')

        sines = powers * torch.sin(FT * np.pi)
        print(f'{sines.shape=}')

        return sines.sum(dim=1) # summing harmonics
