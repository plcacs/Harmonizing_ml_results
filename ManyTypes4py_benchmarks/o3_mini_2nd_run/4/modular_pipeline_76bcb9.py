from __future__ import annotations
import copy
import difflib
from typing import Any, Dict, Iterable, List, Optional, Set, Union
from kedro.pipeline.pipeline import Pipeline
from .transcoding import TRANSCODING_SEPARATOR, _strip_transcoding, _transcode_split
if TYPE_CHECKING:
    from collections.abc import Iterable as CabcIterable
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


def _validate_inputs_outputs(
    inputs: Iterable[str], outputs: Iterable[str], pipe: Pipeline
) -> None:
    inputs = {_strip_transcoding(k) for k in inputs}
    outputs = {_strip_transcoding(k) for k in outputs}
    if any((_is_parameter(i) for i in inputs)):
        raise ModularPipelineError("Parameters should be specified in the 'parameters' argument")
    free_inputs = {_strip_transcoding(i) for i in pipe.inputs()}
    if not inputs <= free_inputs:
        raise ModularPipelineError('Inputs must not be outputs from another node in the same pipeline')
    if outputs & free_inputs:
        raise ModularPipelineError('All outputs must be generated by some node within the pipeline')


def _validate_datasets_exist(
    inputs: Iterable[str], outputs: Iterable[str], parameters: Iterable[str], pipe: Pipeline
) -> None:
    inputs = {_strip_transcoding(k) for k in inputs}
    outputs = {_strip_transcoding(k) for k in outputs}
    existing = {_strip_transcoding(ds) for ds in pipe.datasets()}
    non_existent = (inputs | outputs | set(parameters)) - existing
    if non_existent:
        sorted_non_existent = sorted(non_existent)
        possible_matches: List[str] = []
        for non_existent_input in sorted_non_existent:
            possible_matches += difflib.get_close_matches(non_existent_input, existing)
        error_msg = f'Failed to map datasets and/or parameters onto the nodes provided: {", ".join(sorted_non_existent)}'
        suggestions = f' - did you mean one of these instead: {", ".join(possible_matches)}' if possible_matches else ''
        raise ModularPipelineError(error_msg + suggestions)


def _get_dataset_names_mapping(
    names: Optional[Union[str, Dict[str, str], Iterable[str]]] = None
) -> Dict[str, str]:
    if names is None:
        return {}
    if isinstance(names, str):
        return {names: names}
    if isinstance(names, dict):
        return copy.deepcopy(names)
    return {item: item for item in names}


def _normalize_param_name(name: str) -> str:
    return name if name.startswith('params:') else f'params:{name}'


def _get_param_names_mapping(
    names: Optional[Union[str, Dict[str, str], Iterable[str]]] = None
) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for name, new_name in _get_dataset_names_mapping(names).items():
        if _is_all_parameters(name):
            params[name] = name
        else:
            param_name = _normalize_param_name(name)
            param_new_name = _normalize_param_name(new_name)
            params[param_name] = param_new_name
    return params


def pipeline(
    pipe: Union[Pipeline, Iterable[Union[Node, Pipeline]]],
    *,
    inputs: Optional[Union[str, Dict[str, str], Iterable[str]]] = None,
    outputs: Optional[Union[str, Dict[str, str], Iterable[str]]] = None,
    parameters: Optional[Union[str, Dict[str, str], Iterable[str]]] = None,
    tags: Optional[Set[str]] = None,
    namespace: Optional[str] = None
) -> Pipeline:
    if isinstance(pipe, Pipeline):
        pipe = Pipeline([pipe], tags=tags)
    else:
        pipe = Pipeline(pipe, tags=tags)
    if not any([inputs, outputs, parameters, namespace]):
        return pipe
    inputs = _get_dataset_names_mapping(inputs)
    outputs = _get_dataset_names_mapping(outputs)
    parameters = _get_param_names_mapping(parameters)
    _validate_datasets_exist(inputs.keys(), outputs.keys(), parameters.keys(), pipe)
    _validate_inputs_outputs(inputs.keys(), outputs.keys(), pipe)
    mapping: Dict[str, str] = {**inputs, **outputs, **parameters}

    def _prefix_dataset(name: str) -> str:
        return f'{namespace}.{name}'  # type: ignore

    def _prefix_param(name: str) -> str:
        _, param_name = name.split('params:')
        return f'params:{namespace}.{param_name}'  # type: ignore

    def _is_transcode_base_in_mapping(name: str) -> bool:
        base_name, _ = _transcode_split(name)
        return base_name in mapping

    def _map_transcode_base(name: str) -> str:
        base_name, transcode_suffix = _transcode_split(name)
        return TRANSCODING_SEPARATOR.join((mapping[base_name], transcode_suffix))

    def _rename(name: str) -> str:
        rules: List[
            tuple[
                Any,  # predicate: Callable[[str], bool]
                Any   # processor: Callable[[str], str]
            ]
        ] = [
            (lambda n: n in mapping, lambda n: mapping[n]),
            (_is_all_parameters, lambda n: n),
            (_is_transcode_base_in_mapping, _map_transcode_base),
            (lambda n: bool(namespace) and _is_single_parameter(n), _prefix_param),
            (lambda n: bool(namespace), _prefix_dataset)
        ]
        for predicate, processor in rules:
            if predicate(name):
                processor_name = processor(name)
                return processor_name
        return name

    def _process_dataset_names(
        datasets: Optional[Union[str, List[str], Dict[str, str]]]
    ) -> Optional[Union[str, List[str], Dict[str, str]]]:
        if datasets is None:
            return None
        if isinstance(datasets, str):
            return _rename(datasets)
        if isinstance(datasets, list):
            return [_rename(name) for name in datasets]
        if isinstance(datasets, dict):
            return {key: _rename(value) for key, value in datasets.items()}
        raise ValueError(f'Unexpected input {datasets} of type {type(datasets)}')

    def _copy_node(node: Node) -> Node:
        new_namespace: Optional[str] = node.namespace
        if namespace:
            new_namespace = f'{namespace}.{node.namespace}' if node.namespace else namespace
        node_copy = node._copy(
            inputs=_process_dataset_names(node._inputs),
            outputs=_process_dataset_names(node._outputs),
            namespace=new_namespace
        )
        return node_copy

    new_nodes: List[Node] = [_copy_node(n) for n in pipe.nodes]
    return Pipeline(new_nodes, tags=tags)