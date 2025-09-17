import logging
import re
from os import PathLike
from typing import List, Optional, NamedTuple, Sequence, Union, Dict, Any, Iterable, Type, TypeVar

import torch
from torch import autograd
from allennlp.common import Registrable, Lazy, plugins
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import int_to_device
from allennlp.data import Instance, DatasetReader, DatasetReaderInput, Batch
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader
from allennlp.models import Model, Archive, load_archive
from allennlp.nn.util import move_to_device

logger = logging.getLogger(__name__)

class InstanceInfluence(NamedTuple):
    """
    The influence score associated with this training instance.
    """
    instance: Instance
    loss: float
    score: float

class InterpretOutput(NamedTuple):
    """
    The output associated with a single test instance.
    
    The loss corresponding to the `test_instance`.
    
    The top `k` most influential training instances along with their influence score.
    """
    test_instance: Instance
    loss: float
    top_k: List[InstanceInfluence]

class InstanceWithGrads(NamedTuple):
    """
    Wraps a training `Instance` along with its associated loss and gradients.
    
    `InfluenceInterpreter.train_instances` is a list of these objects.
    """
    instance: Instance
    loss: float
    grads: List[torch.Tensor]

T = TypeVar("T", bound="InfluenceInterpreter")

class InfluenceInterpreter(Registrable):
    """
    An `InfluenceInterpreter` interprets an AllenNLP models's outputs by finding the
    training instances that had the most influence on the prediction for each test input.

    See [Understanding Black-box Predictions via Influence Functions]
    (https://api.semanticscholar.org/CorpusID:13193974) for more information.

    Subclasses are required to implement the `_calculate_influence_scores()` method.

    # Parameters

    model : `Model`, required

    train_data_path : `DatasetReaderInput`, required

    train_dataset_reader : `DatasetReader`, required

    test_dataset_reader : `Optional[DatasetReader]`, optional (default = `None`)
        This is the dataset reader to read the test set file. If not provided, the
        `train_dataset_reader` is used.

    train_data_loader : `Lazy[DataLoader]`, optional (default = `Lazy(SimpleDataLoader)`)
        The data loader used to load training instances.

        !!! Note
            This data loader is only used to call `DataLoader.iter_instances()`, so certain
            `DataLoader` settings like `batch_size` will have no effect.

    test_data_loader : `Lazy[DataLoader]`, optional (default = `Lazy(SimpleDataLoader)`)
        The data loader used to load test instances when `interpret_from_file()` is called.

        !!! Note
            Like `train_data_loader`, this data loader is only used to call `DataLoader.iter_instances()`,
            so certain `DataLoader` settings like `batch_size` will have no effect.

    params_to_freeze : `Optional[List[str]]`, optional (default = `None`)
        An optional list of strings, each of which should be a regular expression that matches
        some parameter keys of the model. Any matching parameters will be have `requires_grad`
        set to `False`.

    cuda_device : `int`, optional (default = `-1`)
        The index of GPU device we want to calculate scores on. If not provided, we uses `-1`
        which correspond to using CPU.
    """
    default_implementation: str = 'simple-influence'

    def __init__(
        self,
        model: Model,
        train_data_path: Union[str, PathLike],
        train_dataset_reader: DatasetReader,
        *,
        test_dataset_reader: Optional[DatasetReader] = None,
        train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        params_to_freeze: Optional[List[str]] = None,
        cuda_device: int = -1,
    ) -> None:
        self.model: Model = model
        self.vocab: Any = model.vocab
        self.device: torch.device = int_to_device(cuda_device)
        self._train_data_path: Union[str, PathLike] = train_data_path
        self._train_loader: DataLoader = train_data_loader.construct(reader=train_dataset_reader, data_path=train_data_path, batch_size=1)
        self._train_loader.set_target_device(self.device)
        self._train_loader.index_with(self.vocab)
        self._test_dataset_reader: DatasetReader = test_dataset_reader or train_dataset_reader
        self._lazy_test_data_loader: Lazy[DataLoader] = test_data_loader
        self.model.to(self.device)
        if params_to_freeze is not None:
            for name, param in self.model.named_parameters():
                if any([re.match(pattern, name) for pattern in params_to_freeze]):
                    param.requires_grad = False
        self._used_params: Optional[List[torch.nn.Parameter]] = None
        self._used_param_names: Optional[List[str]] = None
        self._train_instances: Optional[List[InstanceWithGrads]] = None

    @property
    def used_params(self) -> List[torch.nn.Parameter]:
        """
        The parameters of the model that have non-zero gradients after a backwards pass.

        This can be used to gather the corresponding gradients with respect to a loss
        via the `torch.autograd.grad` function.

        !!! Note
            Accessing this property requires calling `self._gather_train_instances_and_compute_gradients()`
            if it hasn't been called yet, which may take several minutes.
        """
        if self._used_params is None:
            self._gather_train_instances_and_compute_gradients()
        assert self._used_params is not None
        return self._used_params

    @property
    def used_param_names(self) -> List[str]:
        """
        The names of the corresponding parameters in `self.used_params`.

        !!! Note
            Accessing this property requires calling `self._gather_train_instances_and_compute_gradients()`
            if it hasn't been called yet, which may take several minutes.
        """
        if self._used_param_names is None:
            self._gather_train_instances_and_compute_gradients()
        assert self._used_param_names is not None
        return self._used_param_names

    @property
    def train_instances(self) -> List[InstanceWithGrads]:
        """
        The training instances along with their corresponding loss and gradients.

        !!! Note
            Accessing this property requires calling `self._gather_train_instances_and_compute_gradients()`
            if it hasn't been called yet, which may take several minutes.
        """
        if self._train_instances is None:
            self._gather_train_instances_and_compute_gradients()
        assert self._train_instances is not None
        return self._train_instances

    @classmethod
    def from_path(
        cls: Type[T],
        archive_path: Union[str, PathLike],
        *,
        interpreter_name: Optional[str] = None,
        train_data_path: Optional[DatasetReaderInput] = None,
        train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        params_to_freeze: Optional[List[str]] = None,
        cuda_device: int = -1,
        import_plugins: bool = True,
        overrides: Union[str, Dict[str, Any]] = '',
        **extras: Any,
    ) -> T:
        """
        Load an `InfluenceInterpreter` from an archive path.

        # Parameters

        archive_path : `Union[str, PathLike]`, required
            The path to the archive file.
        interpreter_name : `Optional[str]`, optional (default = `None`)
            The registered name of the an interpreter class. If not specified,
            the default implementation (`SimpleInfluence`) will be used.
        train_data_path : `Optional[DatasetReaderInput]`, optional (default = `None`)
            If not specified, `train_data_path` will be taken from the archive's config.
        train_data_loader : `Lazy[DataLoader]`, optional (default = `Lazy(SimpleDataLoader)`)
        test_data_loader : `Lazy[DataLoader]`, optional (default = `Lazy(SimpleDataLoader)`)
        params_to_freeze : `Optional[List[str]]`, optional (default = `None`)
        cuda_device : `int`, optional (default = `-1`)
        import_plugins : `bool`, optional (default = `True`)
            If `True`, we attempt to import plugins before loading the `InfluenceInterpreter`.
            This comes with additional overhead, but means you don't need to explicitly
            import the modules that your implementation depends on as long as those modules
            can be found by `allennlp.common.plugins.import_plugins()`.
        overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
            JSON overrides to apply to the unarchived `Params` object.
        **extras : `Any`
            Extra parameters to pass to the interpreter's `__init__()` method.
        """
        if import_plugins:
            plugins.import_plugins()
        archive: Archive = load_archive(archive_path, cuda_device=cuda_device, overrides=overrides)
        return cls.from_archive(
            archive,
            interpreter_name=interpreter_name,
            train_data_path=train_data_path,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            params_to_freeze=params_to_freeze,
            cuda_device=cuda_device,
            **extras,
        )

    @classmethod
    def from_archive(
        cls: Type[T],
        archive: Archive,
        *,
        interpreter_name: Optional[str] = None,
        train_data_path: Optional[DatasetReaderInput] = None,
        train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        params_to_freeze: Optional[List[str]] = None,
        cuda_device: int = -1,
        **extras: Any,
    ) -> T:
        """
        Load an `InfluenceInterpreter` from an `Archive`.

        The other parameters are the same as `.from_path()`.
        """
        interpreter_cls: Type[T] = cls.by_name(interpreter_name or cls.default_implementation)  # type: ignore
        return interpreter_cls(
            model=archive.model,
            train_data_path=train_data_path or archive.config['train_data_path'],
            train_dataset_reader=archive.dataset_reader,
            test_dataset_reader=archive.validation_dataset_reader,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            params_to_freeze=params_to_freeze,
            cuda_device=cuda_device,
            **extras,
        )

    def interpret(self, test_instance: Instance, k: int = 20) -> InterpretOutput:
        """
        Run the influence function scorer on the given instance, returning the top `k`
        most influential train instances with their scores.

        !!! Note
            Test instances should have `targets` so that a loss can be computed.
        """
        outputs: List[InterpretOutput] = self.interpret_instances([test_instance], k=k)
        return outputs[0]

    def interpret_from_file(self, test_data_path: DatasetReaderInput, k: int = 20) -> List[InterpretOutput]:
        """
        Runs `interpret_instances` over the instances read from `test_data_path`.

        !!! Note
            Test instances should have `targets` so that a loss can be computed.
        """
        test_data_loader: DataLoader = self._lazy_test_data_loader.construct(
            reader=self._test_dataset_reader,
            data_path=test_data_path,
            batch_size=1
        )
        test_data_loader.index_with(self.vocab)
        instances: List[Instance] = list(test_data_loader.iter_instances())
        return self.interpret_instances(instances, k=k)

    def interpret_instances(self, test_instances: Sequence[Instance], k: int = 20) -> List[InterpretOutput]:
        """
        Run the influence function scorer on the given instances, returning the top `k`
        most influential train instances for each test instance.

        !!! Note
            Test instances should have `targets` so that a loss can be computed.
        """
        if not self.train_instances:
            raise ValueError(f'No training instances collected from {self._train_data_path}')
        if not self.used_params:
            raise ValueError('Model has no parameters with non-zero gradients')
        outputs: List[InterpretOutput] = []
        for test_idx, test_instance in enumerate(Tqdm.tqdm(test_instances, desc='test instances')):
            test_batch: Batch = Batch([test_instance])
            test_batch.index_instances(self.vocab)
            test_tensor_dict: Dict[str, Any] = move_to_device(test_batch.as_tensor_dict(), self.device)
            self.model.eval()
            self.model.zero_grad()
            test_output_dict: Dict[str, Any] = self.model(**test_tensor_dict)
            test_loss: torch.Tensor = test_output_dict['loss']
            test_loss_float: float = test_loss.detach().item()
            test_grads: List[torch.Tensor] = list(autograd.grad(test_loss, self.used_params))
            assert len(test_grads) == len(self.used_params)
            influence_scores: torch.Tensor = torch.zeros(len(self.train_instances))
            for idx, score in enumerate(self._calculate_influence_scores(test_instance, test_loss_float, test_grads)):
                influence_scores[idx] = score
            top_k_scores, top_k_indices = torch.topk(influence_scores, k)
            top_k: List[InstanceInfluence] = self._gather_instances(top_k_scores, top_k_indices)
            outputs.append(InterpretOutput(test_instance=test_instance, loss=test_loss_float, top_k=top_k))
        return outputs

    def _gather_instances(self, scores: torch.Tensor, indices: torch.Tensor) -> List[InstanceInfluence]:
        outputs: List[InstanceInfluence] = []
        for score, idx in zip(scores, indices):
            instance, loss, _ = self.train_instances[idx]
            outputs.append(InstanceInfluence(instance=instance, loss=loss, score=score.item()))
        return outputs

    def _gather_train_instances_and_compute_gradients(self) -> None:
        logger.info('Gathering training instances and computing gradients. The result will be cached so this only needs to be done once.')
        self._train_instances = []
        self.model.train()
        for instance in Tqdm.tqdm(self._train_loader.iter_instances(), desc='calculating training gradients'):
            batch: Batch = Batch([instance])
            batch.index_instances(self.vocab)
            tensor_dict: Dict[str, Any] = move_to_device(batch.as_tensor_dict(), self.device)
            self.model.zero_grad()
            output_dict: Dict[str, Any] = self.model(**tensor_dict)
            loss: torch.Tensor = output_dict['loss']
            if self._used_params is None or self._used_param_names is None:
                self._used_params = []
                self._used_param_names = []
                loss.backward(retain_graph=True)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._used_params.append(param)
                        self._used_param_names.append(name)
            grads: List[torch.Tensor] = list(autograd.grad(loss, self._used_params))
            assert len(grads) == len(self._used_params)
            self._train_instances.append(InstanceWithGrads(instance=instance, loss=loss.detach().item(), grads=grads))

    def _calculate_influence_scores(
        self, test_instance: Instance, test_loss: float, test_grads: List[torch.Tensor]
    ) -> Iterable[float]:
        """
        Required to be implemented by subclasses.

        Calculates the influence scores of `self.train_instances` with respect to
        the given `test_instance`.
        """
        raise NotImplementedError
