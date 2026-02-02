import numpy as np
import numpy.typing as npt


def freq_from_pitch(
    pitch: np.floating,
    *,
    A4: np.float32,
) -> np.floating:
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
    frequency: np.floating,
    *,
    A4: np.float32,
) -> np.floating:
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


def generate_waveform_from_spectrum(
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


class SpectrumSynth:
    def __init__(
        self,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
        A4: np.float32,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.A4 = A4

    def gen_sin(
        self,
        pitch: np.floating,
        phase_start: np.floating,
        phase_end: np.floating,
    ) -> npt.NDArray:
        # we only need half the spectrum for real signals
        out = np.zeros(self.buffer_size // 2 + 1, dtype=np.complex64)

        frequency = freq_from_pitch(pitch, A4=self.A4)
        index = frequency * self.buffer_size / self.sample_rate

        floor_index = np.int16(np.floor(index))
        ceil_index = np.int16(np.ceil(index))

        out[floor_index] = ceil_index - index
        out[ceil_index] = index - floor_index

        phases = np.linspace(
            phase_start, phase_end, num=self.buffer_size // 2 + 1
        )

        out = out * np.exp(1j * phases)

        return out
