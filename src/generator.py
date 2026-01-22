import numpy as np
import numpy.typing as npt
import scipy as sp
from src.music import Melody


class Synthesizer:
    def __init__(
        self,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

    def generate_spectrum(
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

        frequency = np.pow(2, (pitch - 69) / 12) * A4

        index = frequency * self.buffer_size / self.sample_rate
        floor_index = np.int16(np.floor(index))
        ceil_index = np.int16(np.ceil(index))

        out[floor_index] = 1.0 * (ceil_index - index)
        out[ceil_index] = 1.0 * (index - floor_index)

        return out

    def gerenate_waveform(self, spectrum: npt.NDArray) -> npt.NDArray:
        complex_audio = np.fft.ifft(spectrum)
        return np.abs(complex_audio)

