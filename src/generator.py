import numpy as np
import numpy.typing as npt
import scipy as sp
from src.music import Melody


def freq_from_pitch(
    pitch: np.floating,
    *,
    A4: np.float32 = np.float32(440),
) -> np.floating:
    return np.pow(2, (pitch - 69) / 12) * A4

def pitch_from_freq(
    frequency: np.floating,
    *,
    A4: np.float32 = np.float32(440),
) -> np.floating:
    return np.log2(frequency / A4) * 12 + 69

class Synthesizer:
    def __init__(
        self,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

    def generate_spectrum_from_pitch(
        self,
        pitch: np.floating,
        *,
        out: npt.NDArray | None = None,
        A4: np.float32 = np.float32(440),
    ) -> npt.NDArray:
        if not out:
            out = np.zeros(self.buffer_size, dtype=np.complex64)
        else:
            out[:] = 0

        frequency = freq_from_pitch(pitch)

        index = frequency * self.buffer_size / self.sample_rate
        floor_index = np.int16(np.floor(index))
        ceil_index = np.int16(np.ceil(index))

        out[floor_index] = 1.0 * (ceil_index - index)
        out[ceil_index] = 1.0 * (index - floor_index)

        out[self.buffer_size - floor_index - 1] = np.conjugate(out[floor_index])
        out[self.buffer_size - ceil_index - 1] = -np.conjugate(out[ceil_index])

        return out

    def gerenate_waveform_from_spectrum(self, spectrum: npt.NDArray) -> npt.NDArray:
        complex_audio = np.fft.ifft(spectrum)
        return complex_audio

