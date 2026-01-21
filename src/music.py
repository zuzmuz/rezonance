from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Note:
    pitch: npt.float32
    start: npt.float32
    duration: npt.float32


class Melody:
    def __init__(self, notes: list[Note]):
        self.notes = np.array(
            [
                np.array([note.pitch, note.start, note.duration])
                for note in notes
            ]
        ).T

    def total_duration(self) -> npt.floating:
        return np.max(self.notes[1] + self.notes[2])
