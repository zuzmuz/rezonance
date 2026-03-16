from pathlib import Path
import torch
from torch.nn import Module
from torch.optim import Optimizer
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

    MODEL_KEY = "model_state"
    OPTIMIZER_KEY = "optimizer_state"

    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def load_from_state(self, model_path: Path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint[self.MODEL_KEY])
        self.optimizer.load_state_dict(checkpoint[self.OPTIMIZER_KEY])

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
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        *,
        nb_epoch: int = 100,
        batch_size: int = 512,
        validate_every: int = 10,
        store_history: bool = True,
        log_epochs: int = 5,
        model_path: Path = Path("saved_models", "model.pth"),
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
                shuffle=True,
                generator=torch.Generator(current_device),
            )

        self.train_history = []
        self.validation_history = []

        epoch = 0
        try:
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
                        self.validation_history.append(
                            validation_loss
                        )
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

        except KeyboardInterrupt:
            logger.info("Interrupted — saving current model state...")
        finally:
            torch.save(
                {
                    self.MODEL_KEY: self.model.state_dict(),
                    self.OPTIMIZER_KEY: self.optimizer.state_dict(),
                },
                model_path,
            )
            logger.info(f"Saved to {model_path}")
