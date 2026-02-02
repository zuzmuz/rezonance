import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from torch.utils.data import Dataset, DataLoader

from src.generator import Synthesizer
from src.music import Melody, Note
from src.training import WaveformDataset, MonophonicModel, Trainer


def main():
    buffer_size = np.int16(1024)
    sample_rate = np.float32(16000)

    dataset = WaveformDataset(
        1000,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        seed=42,
    )

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MonophonicModel(buffer_size)
    
    trainer = Trainer(model)


    history = trainer.train(nb_epoch=1000, data_loader=data_loader)


if __name__ == "__main__":
    main()
