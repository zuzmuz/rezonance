import numpy as np
import numpy.typing as npt

from src.utils import freq_from_pitch

class BaseNoise:
    def __init__(
        self,
        noise_func,
    ):
        self.noise_func = noise_func

    def white(self) -> npt.NDArray:
        return self.noise_func()

    def pink(self) -> npt.NDArray:
        return self.filter_noise(
            lambda freq: 1
            / np.where(freq == 0, float("inf"), np.sqrt(freq))
        )

    def brown(self) -> npt.NDArray:
        return self.filter_noise(
            lambda freq: 1
            / np.where(freq == 0, float("inf"), freq)
        )

    def blue(self) -> npt.NDArray:
        return self.filter_noise(lambda freq: np.sqrt(freq))

    def violet(self) -> npt.NDArray:
        return self.filter_noise(lambda freq: freq)

    # def grey(self) -> npt.NDArray:
    #     return self.filter_noise(lambda freq: np.sqrt(freq) / np.where(freq == 0, float("inf"), freq))

    def filter_noise(self, filter_func) -> npt.NDArray:
        noise = self.noise_func()
        noise_fft = np.fft.rfft(noise)
        filtered_freq = filter_func(
            np.fft.rfftfreq(noise.shape[0])
        )
        # normalize filter to preserve power
        filtered_freq /= filtered_freq.std()
        filtered_noise = noise_fft * filtered_freq
        return np.fft.irfft(filtered_noise)


class NoiseSynth:
    """
    A simple noise synthesizer.
    """
    def __init__(
        self,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self.gaussian = BaseNoise(self._gaussian_noise)
        self.uniform = BaseNoise(self._uniform_noise)

    def _gaussian_noise(self):
        return np.random.normal(
            loc=0.0,
            scale=1.0,
            size=self.buffer_size,
        )

    def _uniform_noise(self) -> npt.NDArray:
        return np.random.randn(self.buffer_size)  # type: ignore


class WaveformSynth:
    """
    A simple waveform synthesizer generating sinewaves from pitch and phase.
    Parameters:
        sample_rate (float): the sample rate of the generated waveform
        buffer_size (int): the buffer size of the generated waveform
        A4 (float): the reference frequency of the A4 note
    """

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

    def gen_single(
        self,
        pitch: np.floating,
        phase: np.floating,
    ) -> npt.NDArray:
        """
        Generate sinusoidal waveform
        Parameters:
            pitch (float): the pith number (logarithmic scale) 69 represents A4
            phase (float): the phase `[-1, 1]`
        Returns:
            Sine wave, not normalized, consider scaling with std
        """
        frequency = freq_from_pitch(pitch, A4=self.A4)  # type: ignore
        linspace = np.linspace(
            0,
            self.buffer_size / self.sample_rate,
            num=self.buffer_size,
        )
        return np.sin((frequency * linspace + phase) * np.pi)

    def gen_multiple(
        self,
        params: npt.NDArray,
    ) -> npt.NDArray:
        """
        Generating waveform as sum of sinewaves from a distriution of parameters.
        Parameters:
            params (matrix): representing sinewaves params, size `3 * n`
                n being the number of sines.
                - The first row is the pitch.
                - The second row is the phase.
                - The third row is the power.
        Returns:
            The sum of all sinewaves, the result is not scaled or normalized, consider dividing by std
        """
        linspace = np.linspace(
            0,
            self.buffer_size / self.sample_rate,
            num=self.buffer_size,
        )

        return (
            params[2, None]  # power
            * np.sin(
                (
                    linspace[:, None]
                    @ freq_from_pitch(
                        params[0, None],  # pitch
                        A4=self.A4,
                    )  # frequency
                    + params[1, None]  # phase
                )
                * np.pi
            )
        ).sum(axis=1)  # summing all sines together
