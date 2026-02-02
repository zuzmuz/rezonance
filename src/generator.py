import numpy as np
import numpy.typing as npt


def freq_from_pitch(
    pitch: np.floating,
    *,
    A4: np.float32,
) -> np.floating:
    return np.pow(2, (pitch - 69) / 12) * A4


def pitch_from_freq(
    frequency: np.floating,
    *,
    A4: np.float32,
) -> np.floating:
    return np.log2(frequency / A4) * 12 + 69


class Synthesizer:
    def __init__(
        self,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
        A4: np.float32
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.A4 = A4

    def generate_spectrum_from_pitch(
        self,
        pitch: np.floating,
    ) -> npt.NDArray:
        # we only need half the spectrum for real signals
        out = np.zeros(self.buffer_size//2 + 1, dtype=np.complex64)

        frequency = freq_from_pitch(pitch, A4=self.A4)

        index = frequency * self.buffer_size / self.sample_rate
        floor_index = np.int16(np.floor(index))
        ceil_index = np.int16(np.ceil(index))

        out[floor_index] = (
            (ceil_index - index)
            * np.exp(2j * np.pi * np.random.random(1)[0])
        )
        out[ceil_index] = (
            (index - floor_index)
            * np.exp(2j * np.pi * np.random.random(1)[0])
        )
        return out

    def generate_waveform_from_spectrum(
        self, spectrum: npt.NDArray
    ) -> npt.NDArray:
        audio = np.fft.irfft(spectrum)
        return audio

    def generate_waveform_from_pitch(
        self,
        pitch: np.floating,
    ) -> npt.NDArray:
        frequency = freq_from_pitch(pitch, A4=self.A4)
        linspace = np.linspace(
            0, self.buffer_size / self.sample_rate, self.buffer_size
        )
        return np.sin(linspace * 2 * np.pi * frequency)

    def generate_spectrum_from_waveform(
        self, waveform: npt.NDArray
    ) -> npt.NDArray:
        return np.fft.fft(waveform)
