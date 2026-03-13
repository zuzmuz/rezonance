import numpy as np
import torch
from torch.types import Tensor, Number
from typing import Callable, overload

from src.utils import freq_from_pitch


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


class Timbre:
    """
    A timbre class for generating harmonic distributions from pitch
    Parameters:
        distriution (tensor): the harmonic distribution tensor, shape `(n, 3)`,
            - n, number of harmonics
            - 3:
                - the frequency multiplier
                - the phase
                - the power
    KewordArguments:
        sample_rate (float): the sample rate of the generated waveform, required to limit frequencies generated to the Shannon frequency
        A4 (float): the reference frequency of the A4 note
    """

    def __init__(
        self,
        distriution: Tensor,
        *,
        sample_rate: Number,
        A4: Number = 440,
    ):
        self.sample_rate = sample_rate
        self.distriution = distriution
        self.A4 = A4

    def gen_harmonics(self, pitch: Tensor) -> Tensor:
        """
        Generate harmonics from pitch according to the timbre's distribution.
        Parameters:
            pitch (tensor): the pitch or pitches to generate harmonics for
                if pitch is a scalar, the output shape is `(n, 3)`,
                if pitch is a 1D tensor of shape `(nb_pitches,)`, the output shape is `(nb_pitches, n, 3)`
        Returns:
            The harmonics tensor, shape depending on the input pitch shape
        """
        frequency = freq_from_pitch(pitch, A4=self.A4)
        # here pitch can be a tensor shape `(nb_pitches)`

        harmonics = self.distriution.clone()  # shape `(n, 3)`

        if pitch.ndim == 1:
            harmonics = harmonics.unsqueeze(0).repeat(
                pitch.size(0), 1, 1
            )
            frequency = frequency.unsqueeze(1)
            harmonics[:, :, 0] = frequency * harmonics[:, :, 0]
        elif pitch.ndim == 0:
            harmonics[:, 0] = frequency * harmonics[:, 0]
        else:
            raise ValueError(
                "Pitch should be a scalar or a 1D tensor"
            )
        return harmonics
