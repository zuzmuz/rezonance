import numpy as np
import numpy.typing as npt

from src.utils import freq_from_pitch


class SpectrumSynth:
    def __init__(
        self,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
        A4: np.floating,
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
        out = np.zeros(
            self.buffer_size // 2 + 1, dtype=np.complex64
        )

        frequency = freq_from_pitch(pitch, A4=self.A4)
        index = (
            frequency * self.buffer_size / self.sample_rate
        )

        floor_index = np.int16(np.floor(index))
        ceil_index = np.int16(np.ceil(index))

        out[floor_index] = ceil_index - index
        out[ceil_index] = index - floor_index

        phases = np.linspace(
            phase_start,
            phase_end,
            num=self.buffer_size // 2 + 1,
        )

        out = out * np.exp(1j * phases)

        return out
