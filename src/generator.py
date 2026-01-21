import numpy as np
import numpy.typing as npt
from src.music import Melody


class Synthesizer:
    def __init__(
        self,
        *,
        sample_rate: np.floating,
    ):
        self.sample_rate = sample_rate

    def generate(
        self,
        melody: Melody,
        *,
        A4: np.floating = np.float32(440),
    ) -> npt.DTypeLike:
        total_duration = melody.total_duration()
        linspace = np.expand_dims(
            np.linspace(
                0,
                total_duration,
                int(np.round(total_duration * self.sample_rate)),
            ),
            axis=1,
        )
        signal = np.zeros_like(linspace)

        all_notes = np.sin(linspace @ melody.notes[0:1])
        return all_notes.sum(axis=1)
