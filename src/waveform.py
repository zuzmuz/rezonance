import numpy as np
import numpy.typing as npt

from src.utils import freq_from_pitch


class WaveformSynth:
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

    def gen(
        self,
        pitches: npt.ArrayLike,
        phases: npt.ArrayLike,
    ) -> npt.NDArray:
        if np.ndim(pitches) != np.ndim(phases):
            raise ValueError("Pitch and phase must have the same size")

        if np.ndim(pitches) == 0:
            pitches = [pitches] # type: ignore
            phases = [phases] # type: ignore

        out = np.zeros(self.buffer_size)

        for pitch, phase in zip(pitches, phases): # type: ignore
            frequency = freq_from_pitch(pitch, A4=self.A4)
            linspace = np.linspace(
                0,
                self.buffer_size / self.sample_rate,
                num=self.buffer_size,
            )
            out += np.sin((frequency * linspace + phase) * np.pi)
        return out
