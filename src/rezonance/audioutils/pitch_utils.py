from typing import overload
import numpy as np
import torch
from torch import Tensor
from torch.types import Number

def freq_from_pitch(
    pitch: Tensor,
    *,
    A4: Number,
) -> Tensor:
    """
    Generate a frequency in Hz from a MIDI pitch number.
    0 is C-1 (8.1758 Hz), 69 is A4
    Parameters:
        pitch (number or tensor): The input pitch number as a floating point value
        A4 (float): The reference frequency for A4 in Hz
    Returns:
        The frequency in Hz, same shape as the input pitch
    """
    return torch.pow(2, (pitch - 69) / 12) * A4


@overload
def pitch_from_freq(
    frequency: Number,
    *,
    A4: Number,
) -> Number: ...


@overload
def pitch_from_freq(
    frequency: Tensor,
    *,
    A4: Number,
) -> Tensor: ...


def pitch_from_freq(
    frequency: Number | Tensor,
    *,
    A4: Number,
) -> Number | Tensor:
    """
    Generate a MIDI pitch number from a frequency in Hz.
    0 is C-1 (8.1758 Hz), 69 is A4 (440 Hz)
    Parameters:
        frequency (number or numpy array): The input frequency in Hz
        A4 (float): The reference frequency for A4 in Hz
    Returns:
        The pitch number, same shape as the input frequency, where
        a difference of 1 corresponds to 100 cents
    """
    if isinstance(frequency, torch.Tensor):
        return torch.log2(frequency / A4) * 12 + 69
    else:
        return np.log2(frequency / A4) * 12 + 69


def gen_wav_from_spectrum(
    spectrum: Tensor,
) -> Tensor:
    """
    Generate a time-domain waveform from a frequency-domain spectrum.
    Parameters:
        spectrum (array): The input frequency-domain spectrum, of size `N`
    Returns:
        The audio waveform in the time domain, of size `(N-1)*2`
    """
    audio = torch.fft.irfft(spectrum)
    return audio


def get_rank_of_pitch(
    pitch: Tensor,
    *,
    sample_rate: Number,
    A4: Number,
) -> Tensor:
    """
    Calculate the possible number of harmonics of a given pitch, taking into account the Shanon frequency
    Parameters:
        pitch (tensor): The pitches to calculate rank for
        sample_rate (Number): The sampling rate
        A4 (Number): The reference frequency for A4 in Hz
    Returns:
        The pitch ranks, same shape as the input pitches
    """

    freq = freq_from_pitch(pitch, A4=A4)
    return sample_rate / (2 * freq)


def get_pitch_of_rank(
    rank: Tensor,
    *,
    sample_rate: Number,
    A4: Number,
) -> Tensor:
    """
    Calculate the possible pitch given the number of harmonics, taking into account the Shanon frequency
    Parameters:
        pitch (tensor): The pitches to calculate rank for
        sample_rate (Number): The sampling rate
        A4 (Number): The reference frequency for A4 in Hz
    Returns:
        The pitch ranks, same shape as the input pitches
    """
    freq = sample_rate / (2 * rank)
    return pitch_from_freq(freq, A4=A4)
