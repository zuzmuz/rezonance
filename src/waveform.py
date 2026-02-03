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

    def gen_single(
        self,
        pitch: np.floating,
        phase: np.floating,
    ) -> npt.NDArray:
        """
        Generate sinusoidal waveform
        Parameters:
            - pitch: the pith number (logarithmic scale) 69 represents A4
            - phase: the phase `[-1, 1]`
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
            - params: matrix representing sinewaves params, size `3 * n`
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
        ).sum(axis=1) # summing all sines together
