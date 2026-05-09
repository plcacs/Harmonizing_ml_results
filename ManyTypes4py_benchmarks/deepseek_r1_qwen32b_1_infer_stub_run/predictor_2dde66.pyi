from typing import List, Dict, Any, Optional, Union, Tuple, Generator, Iterator, ClassVar, Type, Sequence
from torch import Tensor, Module, backends
from torch.utils.hooks import RemovableHandle
from allennlp.common import Registrable
from allennlp.data import Instance, DatasetReader, Batch
from allennlp.models import Model
from allennlp.common.util import JsonDict

class Predictor(Registrable):
    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        ...

    def load_line(self, line: str) -> JsonDict:
        ...

    def dump_line(self, outputs: JsonDict) -> str:
        ...

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        ...

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        ...

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...

    def get_interpretable_layer(self) -> Module:
        ...

    def get_interpretable_text_field_embedder(self) -> Module:
        ...

    def _register_embedding_gradient_hooks(self, embedding_gradients: List[Tensor]) -> List[RemovableHandle]:
        ...

    @contextmanager
    def capture_model_internals(self, module_regex: str = '.*') -> Generator[Dict[str, Any], None, None]:
        ...

    def predict_instance(self, instance: Instance) -> JsonDict:
        ...

    def predictions_to_labeled_instances(self, instance: Instance, outputs: JsonDict) -> List[Instance]:
        ...

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        ...

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        ...

    def predict_batch_instance(self, instances: List[Instance]) -> JsonDict:
        ...

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        ...

    @classmethod
    def from_path(cls, archive_path: Union[str, PathLike], predictor_name: Optional[str] = None, cuda_device: int = -1, dataset_reader_to_load: str = 'validation', frozen: bool = True, import_plugins: bool = True, overrides: Union[str, Dict[str, Any]] = '', **kwargs: Any) -> 'Predictor':
        ...

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: Optional[str] = None, dataset_reader_to_load: str = 'validation', frozen: bool = True, extra_args: Optional[Dict[str, Any]] = None) -> 'Predictor':
        ...