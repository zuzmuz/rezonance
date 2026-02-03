    def test_generation(*args, **kwargs):
        sample_rate = np.float32(16_000)
        buffer_size = np.int16(1024)
        A4 = np.float32(440)
        synth = WaveformSynth(
            sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
        )
        plt.figure(figsize=(13, 5))

        waveform1 = 2.0 * synth.gen_single(70, 0.5) + synth.gen_single(
            80, 0
        )
        waveform1 /= waveform1.std()

        plt.subplot(2, 1, 1)
        plt.plot(waveform1, label='singulars')

        waveform2 = synth.gen_multiple(
            np.array(
                [
                    [70, 80],
                    [0.5, 0],
                    [2.0, 1],
                ]
            )
        )
        waveform2 /= waveform2.std()

        plt.plot(waveform2, label='multiples')

        plt.plot(waveform1 - waveform2, label='dif')
        plt.legend()

        waveform1 = (
            10 * synth.gen_single(40, 0.5)
            + 0.5 * synth.gen_single(60, 0)
            + synth.gen_single(80, 0.2)
            + 5 * synth.gen_single(100, 0.6)
            + 3 * synth.gen_single(50, 0.2)
        )
        waveform1 /= waveform1.std()
        plt.subplot(2, 1, 2)
        plt.plot(waveform1, label='singulars')

        waveform2 = synth.gen_multiple(
            np.array(
                [
                    [40, 60, 80, 100, 50],
                    [0.5, 0, 0.2, 0.6, 0.2],
                    [10, 0.5, 1, 5, 3],
                ]
            )
        )
        waveform2 /= waveform2.std()
        plt.plot(waveform2, label='multiples')

        plt.plot(waveform1 - waveform2, label='dif')
        
        plt.legend()
        plt.show()
