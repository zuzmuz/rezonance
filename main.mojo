struct WaveformSynth:

    var sample_rate: Float32
    var buffer_size: Int
    var A4: Float32

    def __init__(
        out self,
        *,
        sample_rate: Float32,
        buffer_size: Int,
        A4: Float32
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.A4 = A4


def main():
    """What does this do."""
    nvidia = std.sys.info.has_nvidia_gpu_accelerator()
    amd = std.sys.info.has_amd_gpu_accelerator()
    apple = std.sys.info.has_apple_gpu_accelerator()
    print("hi nvidia: {}, amd: {}, apple: {}".format(nvidia, amd, apple))
