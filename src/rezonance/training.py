from pathlib import Path
from typing import Callable
import torch
from torch import Tensor, nn, optim
from torch.types import Number
from torch.utils.data import Dataset, DataLoader

from rezonance.logger import logger
from rezonance.utils import current_device


class Trainer:
    """
    Model trainer class, create one with a model, loss function and optimzer.
    Will train model with validation
    Parameters:
        model (Module): the model to train
        criterion (Loss): a loss function
        optimizer (Optimizer): the loss optimizer

    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def _train_one_epoch(self, data_loader: DataLoader) -> Number:
        logger.debug("Training epoch")
        self.model.train()
        total_loss = 0
        for idx, (batch_X, batch_y) in enumerate(data_loader):
            logger.debug(f"Training {idx}/{len(data_loader)}")
            hat_y = self.model(batch_X)
            loss = self.criterion(hat_y, batch_y)

            # adjusting parameters in training phase
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iteration_loss = loss.item()
            total_loss += iteration_loss

            logger.debug(f"Loss {iteration_loss}")

        return total_loss / len(data_loader)

    def _validate_one_epoch(self, data_loader: DataLoader) -> Number:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, (batch_X, batch_y) in enumerate(data_loader):
                logger.debug(f"Training {idx}/{len(data_loader)}")
                hat_y = self.model(batch_X)
                loss = self.criterion(hat_y, batch_y)
                iteration_loss = loss.item()
                total_loss += iteration_loss

                logger.debug(f"Loss {iteration_loss}")
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        *,
        nb_epoch: int = 100,
        batch_size: int = 512,
        validate_every: int = 10,
        store_history: bool = True,
        log_epochs: int = 5,
    ):
        """
        Train model for an ammound of epochs
        Parameters:
            train_dataset: (Dataset),
            validation_dataset: (Dataset | None)
        Keyword Arguments:
            nb_epoch: (int) number of epochs to train the model for (obviously)
            batch_size: (int) the size of the batches of the dataloader
            store_history: (bool) store history if needed for display or whatever
            log_epochs: (int) logs loss at each checkpoint, set to -1 to disable logging (who whould want that)
            model_path: (Path) path to store model in when training is done

        """

        # creating training dataloader
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(current_device),
        )

        # creating validation dataloader id validation dataset is provided
        validation_data_loader = None
        if validation_dataset:
            validation_data_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=4,
            )

        self.train_history = []
        self.validation_history = []

        epoch = 0
        for epoch in range(nb_epoch):
            train_loss = self._train_one_epoch(train_data_loader)
            validation_loss = None
            if (
                validation_data_loader
                and (epoch + 1) % validate_every == 0
            ):
                validation_loss = self._validate_one_epoch(
                    validation_data_loader
                )

            if store_history:
                self.train_history.append(train_loss)
                if validation_loss:
                    self.validation_history.append(validation_loss)
            if log_epochs > 0 and (epoch + 1) % log_epochs == 0:
                logger.info(
                    f"Epoch {epoch + 1}:"
                    + f"\n\tTraining Loss = {train_loss:.5f}"
                    + (
                        f"\n\tValidation Loss = {validation_loss:.5f}"
                        if validation_loss
                        else ""
                    )
                )
