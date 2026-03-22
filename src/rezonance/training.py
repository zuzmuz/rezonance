from pathlib import Path
from typing import Callable, Literal
from numpy import argmax
import torch
from torch import Tensor, nn, optim
from torch.types import Number
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from rezonance.logger import logger
from rezonance.utils import current_device
from rezonance.transforms import OutputTransform


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
        optimizer: optim.Optimizer,
        output_transform: OutputTransform,
    ):
        self.model = model
        self.optimizer = optimizer
        self.output_transform = output_transform

    def _train_one_epoch(
        self, data_loader: DataLoader, log_batch: int = 0
    ) -> Number:
        logger.debug("Training epoch")
        self.model.train()
        total_loss = 0

        for idx, (batch_X, batch_y) in enumerate(data_loader):
            if log_batch and (idx + 1) % log_batch == 0:
                logger.debug(f"Training {idx + 1}/{len(data_loader)}")

            hat_y = self.model(batch_X)
            transformed_output = self.output_transform.forward(
                batch_y
            )

            loss = self.output_transform.criterion(
                hat_y, transformed_output
            )

            # adjusting parameters in training phase
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iteration_loss = loss.item()
            total_loss += iteration_loss

            accuracy = (
                (hat_y.argmax(-1) == transformed_output)
                .float()
                .mean()
            )

            if log_batch and (idx + 1) % log_batch == 0:
                logger.debug(
                    f"Loss {iteration_loss}, Accuracy {accuracy}"
                )

        return total_loss / len(data_loader)

    def _validate_one_epoch(
        self, data_loader: DataLoader, log_batch: int = 0
    ) -> Number:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, (batch_X, batch_y) in enumerate(data_loader):
                if log_batch and (idx + 1) % log_batch == 0:
                    logger.debug(
                        f"Validating {idx + 1}/{len(data_loader)}"
                    )
                hat_y = self.model(batch_X)

                transformed_output = self.output_transform.forward(
                    batch_y
                )

                loss = self.output_transform.criterion(
                    hat_y, transformed_output
                )

                iteration_loss = loss.item()
                total_loss += iteration_loss

                accuracy = (
                    (hat_y.argmax(-1) == transformed_output)
                    .float()
                    .mean()
                )

                if log_batch and (idx + 1) % log_batch == 0:
                    logger.debug(
                        f"Loss {iteration_loss}, Accuracy {accuracy}"
                    )

                total_loss += iteration_loss()

        return total_loss / len(data_loader)

    def overfit_test(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 64,
        nb_epoch: int = 100,
    ):

        # overfit a single batch to make sure everyting is proper

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(current_device),
        )

        batch_X, batch_y = next(iter(data_loader))

        self.model.train()
        for i in range(nb_epoch):
            hat_y = self.model(batch_X)

            # logger.debug(f"{hat_y=} {batch_y=}")

            transformed_output = self.output_transform.forward(
                batch_y
            )

            # logger.debug(f"{transformed_output=} {torch.argmax(hat_y, -1)=}")

            loss = self.output_transform.criterion(
                hat_y, transformed_output
            )


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iteration_loss = loss.item()

            accuracy = (
                (torch.argmax(hat_y, -1) == transformed_output)
                .float()
                .mean()
            )

            logger.debug(
                f"Loss {iteration_loss}, Accuracy {accuracy}"
            )


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
        log_batch: int = 0,
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
            )

        self.train_history = []
        self.validation_history = []

        epoch = 0
        for epoch in range(nb_epoch):
            train_loss = self._train_one_epoch(
                train_data_loader,
                log_batch=log_batch,
            )

            validation_loss = None

            if (
                validation_data_loader
                and (epoch + 1) % validate_every == 0
            ):
                validation_loss = self._validate_one_epoch(
                    validation_data_loader,
                    log_batch=log_batch,
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
