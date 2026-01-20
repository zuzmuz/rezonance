import numpy as np
import numpy.fft as fft
from numpy.typing import ArrayLike


class SpectrumGenerator:
    def __init__(
        self,
        sample_rate: np.float32,
        buffer_size: np.int16,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.spectrum = None

    def generate_spectrum(
        self, pitch, *, A4: np.float32 = np.float32(440)
    ) -> ArrayLike:
        spectrum = np.zeros(self.buffer_size, dtype=np.complex64)

        frequency = A4 * np.pow(2, (pitch - 69) / 12)

        for i in range(20):
            floating_index = (
                frequency * i * self.buffer_size / self.sample_rate
            )
            floor_index = int(np.floor(floating_index))
            ceil_index = int(np.ceil(floating_index))
            spectrum[floor_index] = 1 * (ceil_index - floating_index)
            spectrum[ceil_index] = 1 * (floating_index - floor_index)
            spectrum[self.buffer_size - floor_index - 1] = 1 * (
                ceil_index - floating_index
            )
            spectrum[self.buffer_size - ceil_index - 1] = 1 * (
                floating_index - floor_index
            )
        return spectrum

    def generate_audio(
        self, spectrum: ArrayLike, duration: float
    ) -> ArrayLike:
        return np.tile(
            fft.ifft(spectrum),
            int(
                np.round(
                    duration * self.sample_rate / self.buffer_size
                )
            ),
        )


