import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
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
        device: torch.device,
        data_convertor: PyTorchDataConvertor,
        model_meta_data: Dict[str, Any] = {},
        window_size: int = 1,
        tb_logger: Optional[SummaryWriter] = None,
        **kwargs: Any
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_meta_data = model_meta_data
        self.device = device
        self.n_epochs: int = kwargs.get('n_epochs', 10)
        self.n_steps: Optional[int] = kwargs.get('n_steps', None)
        if self.n_steps is None and (not self.n_epochs):
            raise Exception('Either `n_steps` or `n_epochs` should be set.')
        self.batch_size: int = kwargs.get('batch_size', 64)
        self.data_convertor = data_convertor
        self.window_size = window_size
        self.tb_logger = tb_logger
        self.test_batch_counter: int = 0

    def fit(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]) -> None:
        self.model.train()
        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary['train_features'])
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter = 0
        for _ in range(n_epochs):
            for _, batch_data in enumerate(data_loaders_dictionary['train']):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                loss = self.criterion(yb_pred, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.tb_logger:
                    self.tb_logger.add_scalar('train_loss', loss.item(), batch_counter)
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
            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)
            if self.tb_logger:
                self.tb_logger.add_scalar(f'{split}_loss', loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1
        self.model.train()

    def create_data_loaders_dictionary(
        self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]
    ) -> Dict[str, DataLoader]:
        data_loader_dictionary: Dict[str, DataLoader] = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f'{split}_features'], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f'{split}_labels'], self.device)
            dataset = TensorDataset(x, y)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
            data_loader_dictionary[split] = data_loader
        return data_loader_dictionary

    def calc_n_epochs(self, n_obs: int) -> int:
        if not isinstance(self.n_steps, int):
            raise ValueError('Either `n_steps` or `n_epochs` should be set.')
        n_batches = n_obs // self.batch_size
        n_epochs = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(f'Setting low n_epochs: {n_epochs}. Please consider increasing `n_steps` hyper-parameter.')
        return n_epochs

    def save(self, path: Union[str, Path]) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_meta_data': self.model_meta_data,
            'pytrainer': self
        }, path)

    def load(self, path: Union[str, Path]) -> 'PyTorchModelTrainer':
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> 'PyTorchModelTrainer':
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_meta_data = checkpoint['model_meta_data']
        return self

class PyTorchTransformerTrainer(PyTorchModelTrainer):

    def create_data_loaders_dictionary(
        self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]
    ) -> Dict[str, DataLoader]:
        data_loader_dictionary: Dict[str, DataLoader] = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f'{split}_features'], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f'{split}_labels'], self.device)
            dataset = WindowDataset(x, y, self.window_size)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=0)
            data_loader_dictionary[split] = data_loader
        return data_loader_dictionary
