from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional, TypeVar, ContextManager
import logging
from os import PathLike
import json
import re
from contextlib import contextmanager
import numpy
import torch
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from torch import backends
from allennlp.common import Registrable, plugins
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util

logger: logging.Logger = logging.getLogger(__name__)

class Predictor(Registrable):
    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        ...

    def load_line(self, line: str) -> Any:
        ...

    def dump_line(self, outputs: Any) -> str:
        ...

    def predict_json(self, inputs: Any) -> Any:
        ...

    def json_to_labeled_instances(self, inputs: Any) -> List[Instance]:
        ...

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...

    def get_interpretable_layer(self) -> Any:
        ...

    def get_interpretable_text_field_embedder(self) -> Any:
        ...

    def _register_embedding_gradient_hooks(self, embedding_gradients: List[Tensor]) -> List[RemovableHandle]:
        ...

    @contextmanager
    def capture_model_internals(self, module_regex: str = '.*') -> ContextManager[Dict[int, Dict[str, Any]]]:
        ...

    def predict_instance(self, instance: Instance) -> Any:
        ...

    def predictions_to_labeled_instances(self, instance: Instance, outputs: Any) -> List[Instance]:
        ...

    def _json_to_instance(self, json_dict: Any) -> Instance:
        ...

    def predict_batch_json(self, inputs: List[Any]) -> Any:
        ...

    def predict_batch_instance(self, instances: List[Instance]) -> Any:
        ...

    def _batch_json_to_instances(self, json_dicts: List[Any]) -> List[Instance]:
        ...

    @classmethod
    def from_path(cls, archive_path: Union[str, PathLike], predictor_name: Optional[str] = None, cuda_device: int = -1, dataset_reader_to_load: str = 'validation', frozen: bool = True, import_plugins: bool = True, overrides: Union[str, Dict[str, Any]] = '', **kwargs: Any) -> 'Predictor':
        ...

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: Optional[str] = None, dataset_reader_to_load: str = 'validation', frozen: bool = True, extra_args: Optional[Dict[str, Any]] = None) -> 'Predictor':
        ...
