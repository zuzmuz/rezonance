import torch
from torch.types import Number, Tensor


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


def pitch_from_freq(
    frequency: Tensor,
    *,
    A4: Number,
) -> Tensor:
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
    return torch.log2(frequency / A4) * 12 + 69


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
