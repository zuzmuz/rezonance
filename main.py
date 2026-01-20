import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from src.generator import SpectrumGenerator


def main():
    s = SpectrumGenerator(
        sample_rate=np.float32(16_000),
        buffer_size=np.int16(1024),
    )

    all_audio = []
    for pitch, duration in [
        (40, 0.5),
        (42, 0.5),
        (44, 0.5),
        (45, 0.5),
        (47, 0.5),
        (49, 0.5),
    ]:
        spectrum = s.generate_spectrum(pitch)
        all_audio.append(
            np.real(s.generate_audio(spectrum, duration))
        )

    sd.play(np.concat(all_audio), 16000)
    sd.wait()


if __name__ == "__main__":
    main()
