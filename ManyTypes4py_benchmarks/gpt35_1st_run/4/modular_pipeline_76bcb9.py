from __future__ import annotations
from typing import TYPE_CHECKING, Set, Iterable, Dict, Union

from kedro.pipeline.pipeline import Pipeline
from .transcoding import TRANSCODING_SEPARATOR, _strip_transcoding, _transcode_split
if TYPE_CHECKING:
    from kedro.pipeline.node import Node

class ModularPipelineError(Exception):
    """Raised when a modular pipeline is not adapted and integrated
    appropriately using the helper.
    """
    pass

def _is_all_parameters(name: str) -> bool:
    return name == 'parameters'

def _is_single_parameter(name: str) -> bool:
    return name.startswith('params:')

def _is_parameter(name: str) -> bool:
    return _is_single_parameter(name) or _is_all_parameters(name)

def _validate_inputs_outputs(inputs: Set[str], outputs: Set[str], pipe: Pipeline) -> None:
    ...

def _validate_datasets_exist(inputs: Set[str], outputs: Set[str], parameters: Set[str], pipe: Pipeline) -> None:
    ...

def _get_dataset_names_mapping(names: Union[str, Set[str], Dict[str, str]]) -> Dict[str, str]:
    ...

def _normalize_param_name(name: str) -> str:
    ...

def _get_param_names_mapping(names: Union[str, Set[str], Dict[str, str]]) -> Dict[str, str]:
    ...

def pipeline(pipe: Union[Pipeline, Iterable[Node]], *, inputs: Union[str, Set[str], Dict[str, str], None] = None, outputs: Union[str, Set[str], Dict[str, str], None] = None, parameters: Union[str, Set[str], Dict[str, str], None] = None, tags: Set[str] = None, namespace: str = None) -> Pipeline:
    ...
