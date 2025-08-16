from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Set
from kedro.pipeline.pipeline import Pipeline
from .transcoding import TRANSCODING_SEPARATOR, _strip_transcoding, _transcode_split
if TYPE_CHECKING:
    from kedro.pipeline.node import Node

class ModularPipelineError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

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

def _get_dataset_names_mapping(names=None) -> dict:
    ...

def _normalize_param_name(name: str) -> str:
    ...

def _get_param_names_mapping(names=None) -> dict:
    ...

def pipeline(pipe: Pipeline, *, inputs=None, outputs=None, parameters=None, tags=None, namespace=None) -> Pipeline:
    ...
