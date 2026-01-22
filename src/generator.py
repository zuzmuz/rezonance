import numpy as np
import numpy.typing as npt
import scipy as sp
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
        indices = np.arange(
            int(np.round(total_duration * self.sample_rate))
        )[:, None]
        linspace = indices / self.sample_rate
        signal = np.zeros_like(linspace)
        frequencies = (
            np.power(2, (melody.notes[0, None] - 69) / 12) * A4
        )

        note_indices_start = melody.notes[1, None] * self.sample_rate
        note_indices_end = (
            melody.notes[1, None] + melody.notes[2, None]
        ) * self.sample_rate

        masks = (indices >= note_indices_start) & (
            indices <= note_indices_end
        )

        all_notes = (
            0.5
            * (
                0.7 * np.sin(linspace @ (frequencies * 2 * np.pi))
                + 0.2 * np.sin(linspace @ (2 * frequencies * 2 * np.pi))
                + 0.3 * np.sin(linspace @ (3 * frequencies * 2 * np.pi))
                + 0.1 * np.sin(linspace @ (4 * frequencies * 2 * np.pi))
                + 0.5 * np.sin(linspace @ (5 * frequencies * 2 * np.pi))
                + 0.1 * np.sin(linspace @ (6 * frequencies * 2 * np.pi))
                + 0.05 * np.sin(linspace @ (7 * frequencies * 2 * np.pi))
                + 0.03 * np.sin(linspace @ (8 * frequencies * 2 * np.pi))
                + 0.01 * np.sin(linspace @ (9 * frequencies * 2 * np.pi))
            )
            * masks
        )

        return all_notes.sum(axis=1)

    def synthesize_note(
        self,
        frequency: np.floating,
    ) -> npt.DTypeLike:
        signal = np.sin(2 * np.pi * frequency * t)
        return signal
