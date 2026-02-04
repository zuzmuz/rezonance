import torch
from src.waveform import Timbre

def run(*args, **kwargs):
    sample_rate = 16_000.0
    A4 = 440.0
    timbre = Timbre(
        torch.tensor([
            [1.0, 0, 0.8],
            [2.0, 0.2, 0.5]
        ]),
        sample_rate=sample_rate,
    )
    
    print(f"harmonics output {timbre.gen_harmonics(torch.tensor([60, 70]))}")


