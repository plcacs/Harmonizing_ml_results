#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import torch
from torch import nn, Tensor
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
        criterion: Any,
        device: Union[str, torch.device],
        data_convertor: PyTorchDataConvertor,
        model_meta_data: Dict[str, Any] = {},
        window_size: int = 1,
        tb_logger: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """
        :param model: The PyTorch model to be trained.
        :param optimizer: The optimizer to use for training.
        :param criterion: The loss function to use for training.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        :param init_model: A dictionary containing the initial model/optimizer
            state_dict and model_meta_data saved by self.save() method.
        :param model_meta_data: Additional metadata about the model (optional).
        :param data_convertor: converter from pd.DataFrame to torch.tensor.
        :param n_steps: used to calculate n_epochs. The number of training iterations to run.
            iteration here refers to the number of times optimizer.step() is called.
            ignored if n_epochs is set.
        :param n_epochs: The maximum number batches to use for evaluation.
        :param batch_size: The size of the batches to use during training.
        """
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.criterion: Any = criterion
        self.model_meta_data: Dict[str, Any] = model_meta_data
        self.device: Union[str, torch.device] = device
        self.n_epochs: Optional[int] = kwargs.get('n_epochs', 10)
        self.n_steps: Optional[int] = kwargs.get('n_steps', None)
        if self.n_steps is None and (not self.n_epochs):
            raise Exception('Either `n_steps` or `n_epochs` should be set.')
        self.batch_size: int = kwargs.get('batch_size', 64)
        self.data_convertor: PyTorchDataConvertor = data_convertor
        self.window_size: int = window_size
        self.tb_logger: Optional[Any] = tb_logger
        self.test_batch_counter: int = 0

    def fit(self, data_dictionary: Dict[str, Any], splits: List[str]) -> None:
        """
        Train the model with the provided data.
        
        :param data_dictionary: Dictionary containing training and test data/labels.
        :param splits: List of splits to use in training, e.g. ["train", "test"].
        """
        self.model.train()
        data_loaders_dictionary: Dict[str, DataLoader] = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs: int = len(data_dictionary['train_features'])
        n_epochs: int = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter: int = 0
        for _ in range(n_epochs):
            for _, batch_data in enumerate(data_loaders_dictionary['train']):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred: Tensor = self.model(xb)
                loss: Tensor = self.criterion(yb_pred, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.tb_logger is not None:
                    self.tb_logger.log_scalar('train_loss', loss.item(), batch_counter)
                batch_counter += 1
            if 'test' in splits:
                self.estimate_loss(data_loaders_dictionary, 'test')

    @torch.no_grad()
    def estimate_loss(self, data_loader_dictionary: Dict[str, DataLoader], split: str) -> None:
        self.model.eval()
        for _, batch_data in enumerate(data_loader_dictionary[split]):
            xb, yb = batch_data
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            yb_pred: Tensor = self.model(xb)
            loss: Tensor = self.criterion(yb_pred, yb)
            if self.tb_logger is not None:
                self.tb_logger.log_scalar(f'{split}_loss', loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1
        self.model.train()

    def create_data_loaders_dictionary(self, data_dictionary: Dict[str, Any], splits: List[str]) -> Dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
        
        :param data_dictionary: Dictionary containing data splits.
        :param splits: List of splits to process.
        :return: Dictionary mapping split names to DataLoader instances.
        """
        data_loader_dictionary: Dict[str, DataLoader] = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f'{split}_features'], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f'{split}_labels'], self.device)
            dataset: TensorDataset = TensorDataset(x, y)
            data_loader: DataLoader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                drop_last=True, 
                num_workers=0
            )
            data_loader_dictionary[split] = data_loader
        return data_loader_dictionary

    def calc_n_epochs(self, n_obs: int) -> int:
        """
        Calculates the number of epochs required to reach the maximum number
        of iterations specified in the model training parameters.
        
        :param n_obs: Number of observations.
        :return: Calculated number of epochs.
        """
        if not isinstance(self.n_steps, int):
            raise ValueError('Either `n_steps` or `n_epochs` should be set.')
        n_batches: int = n_obs // self.batch_size
        n_epochs: int = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(f'Setting low n_epochs: {n_epochs}. Please consider increasing `n_steps` hyper-parameter.')
        return n_epochs

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model state_dict, optimizer state_dict, and metadata.
        
        :param path: File path to save the checkpoint.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_meta_data': self.model_meta_data,
            'pytrainer': self
        }, path)

    def load(self, path: Union[str, Path]) -> "PyTorchModelTrainer":
        checkpoint: Dict[str, Any] = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> "PyTorchModelTrainer":
        """
        Load model and optimizer state_dict as well as metadata from a checkpoint.
        
        :param checkpoint: Dictionary checkpoint loaded using torch.load.
        :return: The trainer with loaded state.
        """
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_meta_data = checkpoint['model_meta_data']
        return self

class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    Creating a trainer for the Transformer model.
    """
    def create_data_loaders_dictionary(self, data_dictionary: Dict[str, Any], splits: List[str]) -> Dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader specifically for Transformer models.
        
        :param data_dictionary: Dictionary containing data splits.
        :param splits: List of splits to process.
        :return: Dictionary mapping split names to DataLoader instances.
        """
        data_loader_dictionary: Dict[str, DataLoader] = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f'{split}_features'], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f'{split}_labels'], self.device)
            dataset = WindowDataset(x, y, self.window_size)
            data_loader: DataLoader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=True, 
                num_workers=0
            )
            data_loader_dictionary[split] = data_loader
        return data_loader_dictionary
