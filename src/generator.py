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
        indices = np.arange(int(np.round(total_duration * self.sample_rate)))[:, None]
        linspace = indices / self.sample_rate
        signal = np.zeros_like(linspace)
        frequencies = np.power(2, (melody.notes[0, None] - 69) / 12) * A4
        note_indices_start = melody.notes[1, None] * self.sample_rate
        note_indices_end = (melody.notes[1, None] + melody.notes[2, None]) * self.sample_rate
        masks = (indices >= note_indices_start) & (indices <= note_indices_end)
        all_notes = np.sin(linspace @ (frequencies * 2 * np.pi)) * masks


        return all_notes.sum(axis=1)
