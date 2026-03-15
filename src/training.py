from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.logger import logger
from src.utils import current_device


class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion  # nn.MSELoss()
        self.optimizer = optimizer

    # optim.Adam(self.model.parameters(), lr=0.001)


    def _train_one_epoch(self, data_loader: DataLoader):
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in data_loader:
            hat_y = self.model(batch_X)
            loss = self.criterion(hat_y, batch_y)

            # adjusting parameters in training phase
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def _validate_one_epoch(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                hat_y = self.model(batch_X)
                loss = self.criterion(hat_y, batch_y)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(
        self,
        nb_epoch: int,
        dataset: Dataset,
        store_history: bool = True,
        log_epochs: int = 5,
        model_path: Path = Path("models", "model.pth"),
    ):
        data_loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator(current_device),
        )

        self.train_history = []
        self.validation_history = []

        epoch = 0
        try:
            for epoch in range(nb_epoch):
                train_loss = self._train_one_epoch(data_loader)
                validation_loss = self._validate_one_epoch(
                    data_loader
                )

                if store_history:
                    self.train_history.append(train_loss)
                    self.validation_history.append(validation_loss)
                if log_epochs > 0 and (epoch + 1) % log_epochs == 0:
                    logger.info(
                        f"Epoch {epoch + 1}:"
                        f"\n\tTraining Loss = {train_loss:.5f}"
                        f"\n\tValidation Loss = {validation_loss:.5f}"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted — saving current model state...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                },
                model_path,
            )
            logger.info(f"Saved to {model_path}")
