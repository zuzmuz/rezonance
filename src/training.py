import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.utils import current_device

class FCModel(nn.Module):
    def __init__(self, buffer_size: int):
        super(FCModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        return self.model(X).squeeze(1)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, nb_epoch: int, dataset: Dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator(current_device),
        )
        history = []
        for epoch in range(nb_epoch):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                hat_y = self.model.forward(batch_X)
                # print(f"{batch_X.shape=} {batch_y.shape=} {hat_y.shape=}")
                loss = self.criterion(hat_y, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                # epoch_accuracy += accuracy_score(
                #     batch_y.detach().numpy(),
                #     hat_y.detach().numpy().argmax(axis=1)
                # )
            history.append(epoch_loss / len(data_loader))
            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}: Mean Squared Error = {history[-1]:.5f}"
                )
        return history
