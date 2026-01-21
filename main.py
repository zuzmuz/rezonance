import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from src.generator import Synthesizer
from src.music import Melody, Note


def main():
    melody = Melody(
        [
            Note(60.0, 0.0, 1),
            Note(67.0, 0.5, 1),
            Note(64.0, 1.0, 1),
            Note(69.0, 1.5, 1),
        ]
    )

    synthesizer = Synthesizer(sample_rate=16000.0)

    signal = synthesizer.generate(melody)

    sd.play(signal, samplerate=16000)
    sd.wait()


if __name__ == "__main__":
    main()
