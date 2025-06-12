import logging
import os
from os import PathLike
import re
from typing import Dict, List, Set, Type, Optional, Union, Any
import numpy
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params, remove_keys_from_params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.nn.module import Module
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.regularizers import RegularizerApplicator

logger = logging.getLogger(__name__)
_DEFAULT_WEIGHTS = 'best.th'

class Model(Module, Registrable):
    _warn_for_unseparable_batches: Set[str] = set()
    default_predictor: Optional[str] = None

    def __init__(self, vocab: Vocabulary, regularizer: Optional[RegularizerApplicator] = None, 
                 serialization_dir: Optional[str] = None, ddp_accelerator: Optional[DdpAccelerator] = None) -> None:
        super().__init__()
        self.vocab = vocab
        self._regularizer = regularizer
        self.serialization_dir = serialization_dir
        self.ddp_accelerator = ddp_accelerator

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        if self._regularizer is None:
            regularization_penalty = None
        else:
            try:
                regularization_penalty = self._regularizer(self)
                if isinstance(regularization_penalty, float):
                    assert regularization_penalty == 0.0
                    regularization_penalty = torch.tensor(regularization_penalty)
            except AssertionError:
                raise RuntimeError('The regularizer cannot be a non-zero float.')
        return regularization_penalty

    def get_parameters_for_histogram_logging(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]

    def get_parameters_for_histogram_tensorboard_logging(self) -> List[str]:
        import warnings
        warnings.warn("'Model.get_parameters_for_histogram_tensorboard_logging' is deprecated, please use 'Model.get_parameters_for_histogram_logging' instead.", DeprecationWarning)
        return self.get_parameters_for_histogram_logging()

    def forward(self, *inputs: Any) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_on_instance(self, instance: Instance) -> Dict[str, Any]:
        return self.forward_on_instances([instance])[0]

    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, Any]]:
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.make_output_human_readable(self(**model_input))
            instance_separated_output = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}

    def _get_prediction_device(self) -> int:
        devices = {util.get_device_of(param) for param in self.parameters()}
        if len(devices) > 1:
            devices_string = ', '.join((str(x) for x in devices))
            raise ConfigurationError(f'Parameters have mismatching cuda_devices: {devices_string}')
        elif len(devices) == 1:
            return devices.pop()
        else:
            return -1

    def _maybe_warn_for_unseparable_batches(self, output_key: str) -> None:
        if output_key not in self._warn_for_unseparable_batches:
            logger.warning(f"Encountered the {output_key} key in the model's return dictionary which couldn't be split by the batch size. Key will be ignored.")
            self._warn_for_unseparable_batches.add(output_key)

    @classmethod
    def _load(cls, config: Params, serialization_dir: str, weights_file: Optional[str] = None, cuda_device: int = -1) -> 'Model':
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)
        vocab_dir = os.path.join(serialization_dir, 'vocabulary')
        vocab_params = config.get('vocabulary', Params({}))
        vocab_choice = vocab_params.pop_choice('type', Vocabulary.list_available(), True)
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(vocab_dir, vocab_params.get('padding_token'), vocab_params.get('oov_token'))
        model_params = config.get('model')
        remove_keys_from_params(model_params)
        model = Model.from_params(vocab=vocab, params=model_params, serialization_dir=serialization_dir)
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()
        model.extend_embedder_vocab()
        model_state = util.read_state_dict(weights_file, cuda_device=cuda_device)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        def filter_out_authorized_missing_keys(module: torch.nn.Module, prefix: str = '') -> None:
            nonlocal missing_keys
            for pat in getattr(module.__class__, 'authorized_missing_keys', None) or []:
                missing_keys = [k for k in missing_keys if k.startswith(prefix) and re.search(pat[len(prefix):], k) is None]
            for name, child in module._modules.items():
                if child is not None:
                    filter_out_authorized_missing_keys(child, prefix + name + '.')
        filter_out_authorized_missing_keys(model)
        if unexpected_keys or missing_keys:
            raise RuntimeError(f'Error loading state dict for {model.__class__.__name__}\n\tMissing keys: {missing_keys}\n\tUnexpected keys: {unexpected_keys}')
        return model

    @classmethod
    def load(cls, config: Params, serialization_dir: str, weights_file: Optional[str] = None, cuda_device: int = -1) -> 'Model':
        model_type = config['model'] if isinstance(config['model'], str) else config['model']['type']
        model_class = cls.by_name(model_type)
        if not isinstance(model_class, type):
            model_class = Model
        return model_class._load(config, serialization_dir, weights_file, cuda_device)

    def extend_embedder_vocab(self, embedding_sources_mapping: Optional[Dict[str, str]] = None) -> None:
        embedding_sources_mapping = embedding_sources_mapping or {}
        for model_path, module in self.named_modules():
            if hasattr(module, 'extend_vocab'):
                pretrained_file = embedding_sources_mapping.get(model_path)
                module.extend_vocab(self.vocab, extension_pretrained_file=pretrained_file, model_path=model_path)

    @classmethod
    def from_archive(cls, archive_file: Union[str, PathLike], vocab: Optional[Vocabulary] = None) -> 'Model':
        from allennlp.models.archival import load_archive
        model = load_archive(archive_file).model
        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()
        return model

Model.register('from_archive', constructor='from_archive')(Model)

def remove_weights_related_keys_from_params(params: Params, keys: List[str] = ['pretrained_file', 'initializer']) -> None:
    remove_keys_from_params(params, keys)

def remove_pretrained_embedding_params(params: Params) -> None:
    remove_keys_from_params(params, ['pretrained_file'])
