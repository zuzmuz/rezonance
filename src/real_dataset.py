import time
from pathlib import Path
import pandas as pd

import torch
from torch import Tensor
from torch.types import Number
from torch.utils.data import Dataset

import torchaudio
from torchcodec.decoders import AudioDecoder

from src.logger import logger


class NSynthDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        sample_rate: Number,
        buffer_size: int,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self.folder = folder
        json_file = folder / "examples.json"

        self.data = pd.read_json(json_file).T.reset_index()

        self.data: pd.DataFrame = self.data[  # type: ignore
            self.data["sample_rate"] == self.sample_rate
        ]

        c = time.perf_counter()

        def get_file_num_frames(row):
            file_name = self.folder / "audio" / f"{row['note_str']}.wav"
            signal, _ = torchaudio.load(file_name)
            return signal.shape

        # def get_file_num_frames(row):
        #     file_name = (
        #         self.folder / "audio" / f"{row['note_str']}.wav"
        #     )
        #     decoded = AudioDecoder(file_name)
        #     return (
        #         decoded.metadata.num_channels,
        #         decoded.metadata.duration_seconds
        #         * decoded.metadata.sample_rate,
        #     )

        self.data["num_frames"] = self.data.apply(
            get_file_num_frames, axis=1
        )

        logger.debug(
            f"{self.data[['note_str', 'num_frames']].head(5)}"
        )
        logger.debug(f"it took {time.perf_counter() - c}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        logger.debug(self.data["note_str"].iloc[idx])
        pitch = self.data["pitch"].iloc[idx]
        file_name = (
            self.folder
            / "audio"
            / f"{self.data['note_str'].iloc[idx]}.wav"
        )

        signal, _ = torchaudio.load(file_name)

        return signal, pitch
