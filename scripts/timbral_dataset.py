import matplotlib.pyplot as plt

from src.dataset import TimbralWaveformDataset

def run(*args, **kwargs):
    sample_rate = 16_000.0
    buffer_size = 1024
    A4 = 440.0
