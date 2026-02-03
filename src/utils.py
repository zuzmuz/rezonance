import numpy as np
import numpy.typing as npt


def freq_from_pitch(
    pitch: npt.NDArray,
    *,
    A4: np.floating,
) -> npt.NDArray:
    """
    Generate a frequency in Hz from a MIDI pitch number.
    0 is C-1 (8.1758 Hz), 69 is A4
    Parameters:
        - pitch: The input pitch number as a floating point value
        - A4: The reference frequency for A4 in Hz
    Returns:
        The frequency in Hz
    """
    return np.pow(2, (pitch - 69) / 12) * A4


def pitch_from_freq(
    frequency: npt.NDArray,
    *,
    A4: np.floating,
) -> npt.NDArray:
    """
    Generate a MIDI pitch number from a frequency in Hz.
    0 is C-1 (8.1758 Hz), 69 is A4 (440 Hz)
    Parameters:
        - frequency: The input frequency in Hz
        - A4: The reference frequency for A4 in Hz
    Returns:
        The pitch number as a floating point value,
        a difference of 1 corresponds to 100 cents
    """
    return np.log2(frequency / A4) * 12 + 69


def gen_wav_from_spectrum(
    spectrum: npt.NDArray,
) -> npt.NDArray:
    """
    Generate a time-domain waveform from a frequency-domain spectrum.
    Parameters:
        - spectrum: The input frequency-domain spectrum, of size `N`
    Returns:
        The audio waveform in the time domain, of size `(N-1)*2`
    """
    audio = np.fft.irfft(spectrum)
    return audio

def gen_sin_from_pitch(
    pitch: np.floatin,
    phase: np.floating,
    size: np.int16,
) -> npt.NDArray:
    """
    Generate a time-domain waveform of a sine wave from a pitch and a phase
    Parameters:
        - pitch: The input pitch number as a floating point value
        - phase: The phase offset, value between `[-1, 1]`
    Rerturns:
        Array containing the sine wave samples
    """
    return np.sin(np.


