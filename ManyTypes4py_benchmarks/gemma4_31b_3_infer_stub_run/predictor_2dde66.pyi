from typing import List, Dict, Tuple, Any, Union, Optional, Iterator, Generator, Type, TypeVar
from os import PathLike
import numpy
import torch
from torch import Tensor
from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive

T = TypeVar("T", bound="Predictor")

class Predictor(Registrable):
    _model: Model
    _dataset_reader: DatasetReader
    cuda_device: int
    _token_offsets: List[Tensor]

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None: ...

    def load_line(self, line: str) -> Any: ...

    def dump_line(self, outputs: Any) -> str: ...

    def predict_json(self, inputs: JsonDict) -> Any: ...

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]: ...

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, numpy.ndarray], Dict[str, Any]]: ...

    def get_interpretable_layer(self) -> torch.nn.Module: ...

    def get_interpretable_text_field_embedder(self) -> torch.nn.Module: ...

    def _register_embedding_gradient_hooks(self, embedding_gradients: List[Tensor]) -> List[torch.utils.hooks.RemovableHandle]: ...

    def capture_model_internals(self, module_regex: str = '.*') -> Generator[Dict[int, Dict[str, Any]], None, None]: ...

    def predict_instance(self, instance: Instance) -> Any: ...

    def predictions_to_labeled_instances(self, instance: Instance, outputs: Any) -> List[Instance]: ...

    def _json_to_instance(self, json_dict: JsonDict) -> Instance: ...

    def predict_batch_json(self, inputs: List[JsonDict]) -> Any: ...

    def predict_batch_instance(self, instances: List[Instance]) -> Any: ...

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]: ...

    @classmethod
    def from_path(
        cls: Type[T],
        archive_path: Union[str, PathLike],
        predictor_name: Optional[str] = None,
        cuda_device: int = -1,
        dataset_reader_to_load: str = 'validation',
        frozen: bool = True,
        import_plugins: bool = True,
        overrides: Union[str, Dict[str, Any]] = '',
        **kwargs: Any
    ) -> T: ...

    @classmethod
    def from_archive(
        cls: Type[T],
        archive: Archive,
        predictor_name: Optional[str] = None,
        dataset_reader_to_load: str = 'validation',
        frozen: bool = True,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> T: ...