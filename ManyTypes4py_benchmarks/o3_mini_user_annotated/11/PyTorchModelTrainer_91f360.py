import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface

from .datasets import WindowDataset

logger = logging.getLogger(__name__)


class PyTorchModelTrainer(PyTorchTrainerInterface):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        data_convertor: PyTorchDataConvertor,
        model_meta_data: Dict[str, Any] = {},
        window_size: int = 1,
        tb_logger: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        :param model: The PyTorch model to be trained.
        :param optimizer: The optimizer to use for training.
        :param criterion: The loss function to use for training.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        :param model_meta_data: Additional metadata about the model (optional).
        :param data_convertor: Converter from pd.DataFrame to torch.Tensor.
        :param window_size: Window size for the dataset.
        :param tb_logger: Logger for tensorboard.
        :param kwargs: Additional keyword arguments.
        """
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.criterion: nn.Module = criterion
        self.model_meta_data: Dict[str, Any] = model_meta_data
        self.device: str = device
        self.n_epochs: int | None = kwargs.get("n_epochs", 10)
        self.n_steps: int | None = kwargs.get("n_steps", None)
        if self.n_steps is None and not self.n_epochs:
            raise Exception("Either `n_steps` or `n_epochs` should be set.")

        self.batch_size: int = kwargs.get("batch_size", 64)
        self.data_convertor: PyTorchDataConvertor = data_convertor
        self.window_size: int = window_size
        self.tb_logger: Any = tb_logger
        self.test_batch_counter: int = 0

    def fit(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]) -> None:
        """
        Trains the model using the provided data.

        :param data_dictionary: A dictionary holding training and test data/labels.
        :param splits: List of splits to use in training. Must contain "train" and can contain "test".
        """
        self.model.train()

        data_loaders_dictionary: Dict[str, DataLoader] = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs: int = len(data_dictionary["train_features"])
        n_epochs: int = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter: int = 0

        for _ in range(n_epochs):
            for _, batch_data in enumerate(data_loaders_dictionary["train"]):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                loss = self.criterion(yb_pred, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.tb_logger is not None:
                    self.tb_logger.log_scalar("train_loss", loss.item(), batch_counter)
                batch_counter += 1

            # evaluation
            if "test" in splits:
                self.estimate_loss(data_loaders_dictionary, "test")

    @torch.no_grad()
    def estimate_loss(
        self,
        data_loader_dictionary: Dict[str, DataLoader],
        split: str,
    ) -> None:
        """
        Estimates and logs the loss on the given split using the provided data loader dictionary.

        :param data_loader_dictionary: A dictionary mapping split names to DataLoaders.
        :param split: The split to evaluate (e.g. "test").
        """
        self.model.eval()
        for _, batch_data in enumerate(data_loader_dictionary[split]):
            xb, yb = batch_data
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)
            if self.tb_logger is not None:
                self.tb_logger.log_scalar(f"{split}_loss", loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1

        self.model.train()

    def create_data_loaders_dictionary(
        self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]
    ) -> Dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.

        :param data_dictionary: A dictionary containing the features and labels for each split.
        :param splits: List of splits to process.
        :return: A dictionary mapping each split to a DataLoader.
        """
        data_loader_dictionary: Dict[str, DataLoader] = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset: TensorDataset = TensorDataset(x, y)
            data_loader: DataLoader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary

    def calc_n_epochs(self, n_obs: int) -> int:
        """
        Calculates the number of epochs required to reach the maximum number
        of iterations specified in the model training parameters.

        :param n_obs: Number of observations in the training dataset.
        :return: Calculated number of epochs.
        """
        if not isinstance(self.n_steps, int):
            raise ValueError("Either `n_steps` or `n_epochs` should be set.")
        n_batches: int = n_obs // self.batch_size
        n_epochs: int = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(
                f"Setting low n_epochs: {n_epochs}. "
                f"Please consider increasing `n_steps` hyper-parameter."
            )

        return n_epochs

    def save(self, path: Path) -> None:
        """
        Saves the model state dictionary, optimizer state, model metadata, and trainer.

        :param path: File path where the model is saved.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_meta_data": self.model_meta_data,
                "pytrainer": self,
            },
            path,
        )

    def load(self, path: Path) -> "PyTorchModelTrainer":
        """
        Loads a saved checkpoint from the specified path.

        :param path: File path of the saved checkpoint.
        :return: The trainer with the loaded checkpoint.
        """
        checkpoint: Dict[str, Any] = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> "PyTorchModelTrainer":
        """
        Loads the state dictionaries and model metadata from a checkpoint.

        :param checkpoint: Checkpoint dictionary containing state dictionaries and metadata.
        :return: The trainer with the loaded state.
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_meta_data = checkpoint["model_meta_data"]
        return self


class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    Trainer for the Transformer model.
    """

    def create_data_loaders_dictionary(
        self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]
    ) -> Dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a custom window dataset for the Transformer model.

        :param data_dictionary: A dictionary containing the features and labels for each split.
        :param splits: List of splits to process.
        :return: A dictionary mapping each split to a DataLoader.
        """
        data_loader_dictionary: Dict[str, DataLoader] = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset: WindowDataset = WindowDataset(x, y, self.window_size)
            data_loader: DataLoader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary