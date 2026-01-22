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

    synth = Synthesizer(
        sample_rate=np.float32(16000),
        buffer_size=np.int16(1024)
    )
    # spectrum = synth.generate_spectrum_from_pitch(np.float32(72))
    # audio = synth.gerenate_waveform_from_spectrum(spectrum)
    #
    # plt.plot(np.real(audio))
    # plt.plot(np.imag(audio))
    # plt.plot(np.abs(audio))
    # plt.show()

    audio = synth.generate_waveform_from_pitch(np.float32(69))
    spectrum = synth.generate_spectrum_from_waveform(audio)
    regen_audio = synth.generate_waveform_from_spectrum(spectrum)
    spectrum = synth.generate_spectrum_from_pitch(np.float32(69))
    gen_audio = synth.generate_waveform_from_spectrum(spectrum)


    plt.plot(np.abs(gen_audio))
    # plt.plot(np.imag(gen_audio))
    plt.show()
    #
    # sd.play(signal, samplerate=16000)
    # sd.wait()


if __name__ == "__main__":
    main()
