import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.types import Number, Tensor


logger = logging.getLogger(__name__)


class LinearModel1(nn.Module):
    def __init__(self, buffer_size: int):
        super(LinearModel1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        return self.model(X)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, nb_epoch: int, dataset: Dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator(
                device=torch.get_default_device()
            ),
        )

        perf_counter = time.perf_counter()
        for epoch in range(nb_epoch):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                hat_y = self.model.forward(batch_X).squeeze()
                loss = self.criterion(hat_y, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                num_batches = len(data_loader)
                print(
                    f"Epoch {epoch + 1}: "
                    f"Mean Squared Error = {epoch_loss / num_batches:.5f}, "
                    f"Time = {time.perf_counter() - perf_counter:.2f} seconds"
                )
                perf_counter = time.perf_counter()
        return []

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
