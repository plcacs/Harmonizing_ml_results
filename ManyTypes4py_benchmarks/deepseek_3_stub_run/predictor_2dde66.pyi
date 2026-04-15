from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional, ContextManager
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

logger: logging.Logger = ...

class Predictor(Registrable):
    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        frozen: bool = True
    ) -> None: ...
    
    def load_line(self, line: str) -> Any: ...
    
    def dump_line(self, outputs: Any) -> str: ...
    
    def predict_json(self, inputs: JsonDict) -> JsonDict: ...
    
    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]: ...
    
    def get_gradients(
        self,
        instances: List[Instance]
    ) -> Tuple[Dict[str, numpy.ndarray], JsonDict]: ...
    
    def get_interpretable_layer(self) -> torch.nn.Module: ...
    
    def get_interpretable_text_field_embedder(self) -> torch.nn.Module: ...
    
    def _register_embedding_gradient_hooks(
        self,
        embedding_gradients: List[Tensor]
    ) -> List[RemovableHandle]: ...
    
    def capture_model_internals(
        self,
        module_regex: str = '.*'
    ) -> ContextManager[Dict[int, Dict[str, Any]]]: ...
    
    def predict_instance(self, instance: Instance) -> JsonDict: ...
    
    def predictions_to_labeled_instances(
        self,
        instance: Instance,
        outputs: JsonDict
    ) -> List[Instance]: ...
    
    def _json_to_instance(self, json_dict: JsonDict) -> Instance: ...
    
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]: ...
    
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]: ...
    
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]: ...
    
    @classmethod
    def from_path(
        cls,
        archive_path: Union[str, PathLike],
        predictor_name: Optional[str] = None,
        cuda_device: int = -1,
        dataset_reader_to_load: str = 'validation',
        frozen: bool = True,
        import_plugins: bool = True,
        overrides: Union[str, Dict[str, Any]] = '',
        **kwargs: Any
    ) -> 'Predictor': ...
    
    @classmethod
    def from_archive(
        cls,
        archive: Archive,
        predictor_name: Optional[str] = None,
        dataset_reader_to_load: str = 'validation',
        frozen: bool = True,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> 'Predictor': ...
    
    _model: Model
    _dataset_reader: DatasetReader
    cuda_device: int
    _token_offsets: List[Tensor]