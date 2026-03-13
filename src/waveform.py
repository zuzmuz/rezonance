import numpy as np
import torch
from torch.types import Tensor, Number
from typing import Callable, overload

from src.utils import freq_from_pitch, get_rank_of_pitch


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
                    - The frequency.
                    - The phase.
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
        frequencies = params[:, 0].unsqueeze(1)
        phases = params[:, 1].unsqueeze(1)

        return torch.sin((phases + frequencies @ linspace) * np.pi)

    def gen_poly(
        self,
        frequencies: Tensor,
        phases: Tensor,
        powers: Tensor,
    ) -> Tensor:

        frequencies = frequencies.unsqueeze(2)
        phases = phases.unsqueeze(2)
        powers = powers.unsqueeze(2)

        # Generating the linspace for the waveform, this will represent time
        linspace = (
            torch.linspace(
                0,
                self.buffer_size / self.sample_rate,
                self.buffer_size,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(frequencies.size(0), 1, 1)
        )
        # adding a dimension and repeating for each signal,
        # linspace shape `(n, 1, buffer_size)`

        FT = torch.baddbmm(phases, frequencies, linspace)

        sines = powers * torch.sin(FT * np.pi)

        return sines.sum(dim=1)  # summing harmonics

    def gen_poly_from_params(self, params: Tensor) -> Tensor:
        """
        Generating polyphonic waveform as sum of sinewaves from a distriution of parameters.
        Parameters:
            params (tensor): representing sinewaves params, size `(n, p, 3)`
                - n being the number of signals required.
                - p the number of harmonics per signal.
                - 3:
                    - The frequency.
                    - The phase.
                    - The power.
        Returns:
            A tensor containing all signals with the sum of all harmonics,
            of shape `(n, buffer_size)`,
            the result is not scaled or normalized, consider dividing by std
        """

        # Creating frequencies and phases matrices from params, shape(n, p,)
        frequencies = params[:, :, 0]
        phases = params[:, :, 1]
        powers = params[:, :, 2]

        return self.gen_poly(
            frequencies,
            phases,
            powers,
        )


class InstrumentSynth:
    def __init__(
        self,
        power_dist: Callable[[Tensor, int], Tensor],
        phase_dist: Callable[[Tensor, int], Tensor],
        *,
        buffer_size: int,
        sample_rate: Number,
        A4: Number,
    ):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.A4 = A4
        self.synth = WaveformSynth(
            sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
        )
        self.power_dist = power_dist
        self.phase_dist = phase_dist

    def __call__(
        self, pitches: Tensor, *, per_pitch: int = 1
    ) -> Tensor:
        freqs = freq_from_pitch(pitches, A4=self.A4)
        ranks = get_rank_of_pitch(
            pitches, sample_rate=self.sample_rate, A4=self.A4
        ).floor()

        signals = torch.zeros(
            pitches.size(0) * per_pitch, self.buffer_size
        )

        for idx, (rank, pitch) in enumerate(zip(ranks, pitches)):
            multipliers = torch.arange(
                1, int(rank + 1)
            )  # this is not efficient, rank can stay the same
            freqs = freq_from_pitch(pitch, A4=self.A4) * multipliers
            powers = self.power_dist(multipliers, per_pitch)
            phases = self.phase_dist(multipliers, per_pitch)

            signal = self.synth.gen_poly(
                freqs.repeat(per_pitch, 1), powers, phases
            )

            signals[idx * per_pitch : (idx + 1) * per_pitch] = signal

        return signals


class Instrument:
    @classmethod
    def saw(
        cls, *, buffer_size: int, sample_rate: Number, A4: Number
    ) -> InstrumentSynth:

        return InstrumentSynth(
            power_dist=lambda multipliers, per_pitch: (
                1 / (multipliers)
            ).repeat(per_pitch, 1),
            phase_dist=lambda multipliers, per_pitch: torch.zeros(
                per_pitch, multipliers.size(0)
            ),
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            A4=A4,
        )

    @classmethod
    def square(
        cls, *, buffer_size: int, sample_rate: Number, A4: Number
    ) -> InstrumentSynth:
        
        def power_dist_func(multipliers: Tensor, per_pitch: int) -> Tensor:
            mask = torch.zeros_like(multipliers)
            mask[::2] = 1
            return (mask / multipliers).repeat(per_pitch, 1)


        return InstrumentSynth(
            power_dist=power_dist_func,
            phase_dist=lambda multipliers, per_pitch: torch.zeros(
                per_pitch, multipliers.size(0)
            ),
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            A4=A4,
        )

