import logging
from os import PathLike
import re
from typing import List, Optional, NamedTuple, Sequence, Union, Dict, Any
import torch
from torch import autograd
from allennlp.common import Registrable, Lazy, plugins
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import int_to_device
from allennlp.data import Instance, DatasetReader, DatasetReaderInput, Batch
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader
from allennlp.models import Model, Archive, load_archive
from allennlp.nn.util import move_to_device

logger: logging.Logger = logging.getLogger(__name__)

class InstanceInfluence(NamedTuple):
    instance: Instance
    loss: float
    score: float

class InterpretOutput(NamedTuple):
    loss: float
    top_k: List[InstanceInfluence]

class InstanceWithGrads(NamedTuple):
    instance: Instance
    loss: float
    grads: List[torch.Tensor]

class InfluenceInterpreter(Registrable):
    default_implementation: str = 'simple-influence'

    def __init__(self, model: Model, train_data_path: DatasetReaderInput, train_dataset_reader: DatasetReader, *, test_dataset_reader: Optional[DatasetReader] = None, train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader), test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader), params_to_freeze: Optional[List[str]] = None, cuda_device: int = -1):
        self.model: Model = model
        self.vocab = model.vocab
        self.device = int_to_device(cuda_device)
        self._train_data_path = train_data_path
        self._train_loader = train_data_loader.construct(reader=train_dataset_reader, data_path=train_data_path, batch_size=1)
        self._train_loader.set_target_device(self.device)
        self._train_loader.index_with(self.vocab)
        self._test_dataset_reader = test_dataset_reader or train_dataset_reader
        self._lazy_test_data_loader = test_data_loader
        self.model.to(self.device)
        if params_to_freeze is not None:
            for name, param in self.model.named_parameters():
                if any([re.match(pattern, name) for pattern in params_to_freeze]):
                    param.requires_grad = False
        self._used_params = None
        self._used_param_names = None
        self._train_instances = None

    @property
    def used_params(self) -> List[torch.Tensor]:
        if self._used_params is None:
            self._gather_train_instances_and_compute_gradients()
        assert self._used_params is not None
        return self._used_params

    @property
    def used_param_names(self) -> List[str]:
        if self._used_param_names is None:
            self._gather_train_instances_and_compute_gradients()
        assert self._used_param_names is not None
        return self._used_param_names

    @property
    def train_instances(self) -> List[InstanceWithGrads]:
        if self._train_instances is None:
            self._gather_train_instances_and_compute_gradients()
        assert self._train_instances is not None
        return self._train_instances

    @classmethod
    def from_path(cls, archive_path: Union[str, PathLike], interpreter_name: Optional[str] = None, train_data_path: Optional[DatasetReaderInput] = None, train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader), test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader), params_to_freeze: Optional[List[str]] = None, cuda_device: int = -1, import_plugins: bool = True, overrides: Union[str, Dict[str, Any]] = '', **extras: Any):
        if import_plugins:
            plugins.import_plugins()
        return cls.from_archive(load_archive(archive_path, cuda_device=cuda_device, overrides=overrides), interpreter_name=interpreter_name, train_data_path=train_data_path, train_data_loader=train_data_loader, test_data_loader=test_data_loader, params_to_freeze=params_to_freeze, cuda_device=cuda_device, **extras)

    @classmethod
    def from_archive(cls, archive: Archive, interpreter_name: Optional[str] = None, train_data_path: Optional[DatasetReaderInput] = None, train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader), test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader), params_to_freeze: Optional[List[str]] = None, cuda_device: int = -1, **extras: Any):
        interpreter_cls = cls.by_name(interpreter_name or cls.default_implementation)
        return interpreter_cls(model=archive.model, train_data_path=train_data_path or archive.config['train_data_path'], train_dataset_reader=archive.dataset_reader, test_dataset_reader=archive.validation_dataset_reader, train_data_loader=train_data_loader, test_data_loader=test_data_loader, params_to_freeze=params_to_freeze, cuda_device=cuda_device, **extras)

    def interpret(self, test_instance: Instance, k: int = 20) -> InterpretOutput:
        return self.interpret_instances([test_instance], k=k)[0]

    def interpret_from_file(self, test_data_path: str, k: int = 20) -> List[InterpretOutput]:
        test_data_loader = self._lazy_test_data_loader.construct(reader=self._test_dataset_reader, data_path=test_data_path, batch_size=1)
        test_data_loader.index_with(self.vocab)
        instances = list(test_data_loader.iter_instances())
        return self.interpret_instances(instances, k=k)

    def interpret_instances(self, test_instances: List[Instance], k: int = 20) -> List[InterpretOutput]:
        if not self.train_instances:
            raise ValueError(f'No training instances collected from {self._train_data_path}')
        if not self.used_params:
            raise ValueError('Model has no parameters with non-zero gradients')
        outputs = []
        for test_idx, test_instance in enumerate(Tqdm.tqdm(test_instances, desc='test instances')):
            test_batch = Batch([test_instance])
            test_batch.index_instances(self.vocab)
            test_tensor_dict = move_to_device(test_batch.as_tensor_dict(), self.device)
            self.model.eval()
            self.model.zero_grad()
            test_output_dict = self.model(**test_tensor_dict)
            test_loss = test_output_dict['loss']
            test_loss_float = test_loss.detach().item()
            test_grads = autograd.grad(test_loss, self.used_params)
            assert len(test_grads) == len(self.used_params)
            influence_scores = torch.zeros(len(self.train_instances))
            for idx, score in enumerate(self._calculate_influence_scores(test_instance, test_loss_float, test_grads)):
                influence_scores[idx] = score
            top_k_scores, top_k_indices = torch.topk(influence_scores, k)
            top_k = self._gather_instances(top_k_scores, top_k_indices)
            outputs.append(InterpretOutput(loss=test_loss_float, top_k=top_k))
        return outputs

    def _gather_instances(self, scores: torch.Tensor, indices: torch.Tensor) -> List[InstanceInfluence]:
        outputs = []
        for score, idx in zip(scores, indices):
            instance, loss, _ = self.train_instances[idx]
            outputs.append(InstanceInfluence(instance=instance, loss=loss, score=score.item()))
        return outputs

    def _gather_train_instances_and_compute_gradients(self):
        logger.info('Gathering training instances and computing gradients. The result will be cached so this only needs to be done once.')
        self._train_instances = []
        self.model.train()
        for instance in Tqdm.tqdm(self._train_loader.iter_instances(), desc='calculating training gradients'):
            batch = Batch([instance])
            batch.index_instances(self.vocab)
            tensor_dict = move_to_device(batch.as_tensor_dict(), self.device)
            self.model.zero_grad()
            output_dict = self.model(**tensor_dict)
            loss = output_dict['loss']
            if self._used_params is None or self._used_param_names is None:
                self._used_params = []
                self._used_param_names = []
                loss.backward(retain_graph=True)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._used_params.append(param)
                        self._used_param_names.append(name)
            grads = autograd.grad(loss, self._used_params)
            assert len(grads) == len(self._used_params)
            self._train_instances.append(InstanceWithGrads(instance=instance, loss=loss.detach().item(), grads=grads))

    def _calculate_influence_scores(self, test_instance: Instance, test_loss: float, test_grads: List[torch.Tensor]) -> List[float]:
        raise NotImplementedError
