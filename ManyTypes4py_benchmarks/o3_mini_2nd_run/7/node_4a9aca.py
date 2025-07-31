from __future__ import annotations
import copy
import inspect
import logging
import re
from collections import Counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from warnings import warn
from more_itertools import spy, unzip
from .transcoding import _strip_transcoding

if TYPE_CHECKING:
    from collections.abc import Iterable as IterableABC


class Node:
    def __init__(
        self,
        func: Callable[..., Any],
        inputs: Optional[Union[str, List[str], Dict[str, str]]] = None,
        outputs: Optional[Union[str, List[str], Dict[str, str]]] = None,
        *,
        name: Optional[str] = None,
        tags: Optional[Union[str, Iterable[str]]] = None,
        confirms: Optional[Union[str, List[str]]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        if not callable(func):
            raise ValueError(_node_error_message(f"first argument must be a function, not '{type(func).__name__}'."))
        if inputs and (not isinstance(inputs, (list, dict, str))):
            raise ValueError(_node_error_message(f"'inputs' type must be one of [String, List, Dict, None], not '{type(inputs).__name__}'."))
        for _input in _to_list(inputs):
            if not isinstance(_input, str):
                raise ValueError(_node_error_message(f"names of variables used as inputs to the function must be of 'String' type, but {_input} from {inputs} is '{type(_input)}'."))
        if outputs and (not isinstance(outputs, (list, dict, str))):
            raise ValueError(_node_error_message(f"'outputs' type must be one of [String, List, Dict, None], not '{type(outputs).__name__}'."))
        for _output in _to_list(outputs):
            if not isinstance(_output, str):
                raise ValueError(_node_error_message(f"names of variables used as outputs of the function must be of 'String' type, but {_output} from {outputs} is '{type(_output)}'."))
        if not inputs and (not outputs):
            raise ValueError(_node_error_message("it must have some 'inputs' or 'outputs'."))
        self._validate_inputs(func, inputs)
        self._func: Callable[..., Any] = func
        self._inputs: Optional[Union[str, List[str], Dict[str, str]]] = inputs
        self._outputs: Optional[Union[str, List[str], Dict[str, str]]] = outputs
        if name and (not re.match(r'[\w\.-]+$', name)):
            raise ValueError(f"'{name}' is not a valid node name. It must contain only letters, digits, hyphens, underscores and/or fullstops.")
        self._name: Optional[str] = name
        self._namespace: Optional[str] = namespace
        self._tags: Set[str] = set(_to_list(tags))
        for tag in self._tags:
            if not re.match(r'[\w\.-]+$', tag):
                raise ValueError(f"'{tag}' is not a valid node tag. It must contain only letters, digits, hyphens, underscores and/or fullstops.")
        self._validate_unique_outputs()
        self._validate_inputs_dif_than_outputs()
        self._confirms: Optional[Union[str, List[str]]] = confirms

    def _copy(self, **overwrite_params: Any) -> Node:
        params: Dict[str, Any] = {
            'func': self._func,
            'inputs': self._inputs,
            'outputs': self._outputs,
            'name': self._name,
            'namespace': self._namespace,
            'tags': self._tags,
            'confirms': self._confirms,
        }
        params.update(overwrite_params)
        return Node(**params)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @property
    def _unique_key(self) -> Tuple[Optional[str], Any, Any]:
        def hashable(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted(value.items()))
            if isinstance(value, list):
                return tuple(value)
            return value
        return (self.name, hashable(self._inputs), hashable(self._outputs))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self._unique_key == other._unique_key

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self._unique_key < other._unique_key

    def __hash__(self) -> int:
        return hash(self._unique_key)

    def __str__(self) -> str:
        def _set_to_str(xset: Iterable[str]) -> str:
            return f"[{';'.join(xset)}]"
        out_str: str = _set_to_str(self.outputs) if self._outputs else 'None'
        in_str: str = _set_to_str(self.inputs) if self._inputs else 'None'
        prefix: str = self._name + ': ' if self._name else ''
        return prefix + f'{self._func_name}({in_str}) -> {out_str}'

    def __repr__(self) -> str:
        return f'Node({self._func_name}, {self._inputs!r}, {self._outputs!r}, {self._name!r})'

    def __call__(self, **kwargs: Any) -> Any:
        return self.run(inputs=kwargs)

    @property
    def _func_name(self) -> str:
        name: str = _get_readable_func_name(self._func)
        if name == '<partial>':
            warn(f"The node producing outputs '{self.outputs}' is made from a 'partial' function. Partial functions do not have a '__name__' attribute: consider using 'functools.update_wrapper' for better log messages.")
        return name

    @property
    def func(self) -> Callable[..., Any]:
        return self._func

    @func.setter
    def func(self, func: Callable[..., Any]) -> None:
        self._func = func

    @property
    def tags(self) -> Set[str]:
        return set(self._tags)

    def tag(self, tags: Union[str, Iterable[str]]) -> Node:
        return self._copy(tags=self.tags | set(_to_list(tags)))

    @property
    def name(self) -> str:
        node_name: str = self._name or str(self)
        if self.namespace:
            return f'{self.namespace}.{node_name}'
        return node_name

    @property
    def short_name(self) -> str:
        if self._name:
            return self._name
        return self._func_name.replace('_', ' ').title()

    @property
    def namespace(self) -> Optional[str]:
        return self._namespace

    @property
    def inputs(self) -> List[str]:
        if isinstance(self._inputs, dict):
            return _dict_inputs_to_list(self._func, self._inputs)
        return _to_list(self._inputs)

    @property
    def outputs(self) -> List[str]:
        return _to_list(self._outputs)

    @property
    def confirms(self) -> List[str]:
        return _to_list(self._confirms)

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._logger.info('Running node: %s', str(self))
        outputs: Any = None
        if not (inputs is None or isinstance(inputs, dict)):
            raise ValueError(f'Node.run() expects a dictionary or None, but got {type(inputs)} instead')
        try:
            inputs = {} if inputs is None else inputs
            if not self._inputs:
                outputs = self._run_with_no_inputs(inputs)
            elif isinstance(self._inputs, str):
                outputs = self._run_with_one_input(inputs, self._inputs)
            elif isinstance(self._inputs, list):
                outputs = self._run_with_list(inputs, self._inputs)
            elif isinstance(self._inputs, dict):
                outputs = self._run_with_dict(inputs, self._inputs)
            return self._outputs_to_dictionary(outputs)
        except Exception as exc:
            self._logger.error('Node %s failed with error: \n%s', str(self), str(exc), extra={'markup': True})
            raise exc

    def _run_with_no_inputs(self, inputs: Dict[str, Any]) -> Any:
        if inputs:
            raise ValueError(f'Node {self!s} expected no inputs, but got the following {len(inputs)} input(s) instead: {sorted(inputs.keys())}.')
        return self._func()

    def _run_with_one_input(self, inputs: Dict[str, Any], node_input: str) -> Any:
        if len(inputs) != 1 or node_input not in inputs:
            raise ValueError(f"Node {self!s} expected one input named '{node_input}', but got the following {len(inputs)} input(s) instead: {sorted(inputs.keys())}.")
        return self._func(inputs[node_input])

    def _run_with_list(self, inputs: Dict[str, Any], node_inputs: List[str]) -> Any:
        if set(node_inputs) != set(inputs.keys()):
            raise ValueError(f'Node {self!s} expected {len(node_inputs)} input(s) {node_inputs}, but got the following {len(inputs)} input(s) instead: {sorted(inputs.keys())}.')
        return self._func(*(inputs[item] for item in node_inputs))

    def _run_with_dict(self, inputs: Dict[str, Any], node_inputs: Dict[str, str]) -> Any:
        if set(node_inputs.values()) != set(inputs.keys()):
            raise ValueError(f'Node {self!s} expected {len(set(node_inputs.values()))} input(s) {sorted(set(node_inputs.values()))}, but got the following {len(inputs)} input(s) instead: {sorted(inputs.keys())}.')
        kwargs: Dict[str, Any] = {arg: inputs[alias] for arg, alias in node_inputs.items()}
        return self._func(**kwargs)

    def _outputs_to_dictionary(self, outputs: Any) -> Dict[str, Any]:
        def _from_dict() -> Dict[str, Any]:
            result, iterator = (outputs, None)
            if inspect.isgenerator(outputs):
                (result,), iterator = spy(outputs)
            keys: List[str] = list(self._outputs.keys())  # type: ignore
            names: List[str] = list(self._outputs.values())  # type: ignore
            if not isinstance(result, dict):
                raise ValueError(f'Failed to save outputs of node {self}.\nThe node output is a dictionary, whereas the function output is {type(result)}.')
            if set(keys) != set(result.keys()):
                raise ValueError(f"Failed to save outputs of node {self!s}.\nThe node's output keys {set(result.keys())} do not match with the returned output's keys {set(keys)}.")
            if iterator:
                exploded = map(lambda x: tuple((x[k] for k in keys)), iterator)
                result_tuple = unzip(exploded)
            else:
                result_tuple = tuple((result[k] for k in keys))
            return dict(zip(names, result_tuple))

        def _from_list() -> Dict[str, Any]:
            result, iterator = (outputs, None)
            if inspect.isgenerator(outputs):
                (result,), iterator = spy(outputs)
            if not isinstance(result, (list, tuple)):
                raise ValueError(f"Failed to save outputs of node {self!s}.\nThe node definition contains a list of outputs {self._outputs}, whereas the node function returned a '{type(result).__name__}'.")
            if len(result) != len(self._outputs):  # type: ignore
                raise ValueError(f'Failed to save outputs of node {self!s}.\nThe node function returned {len(result)} output(s), whereas the node definition contains {len(self._outputs)} output(s).')
            if iterator:
                result_tuple = unzip(iterator)
            else:
                result_tuple = result
            return dict(zip(self._outputs, result_tuple))  # type: ignore

        if self._outputs is None:
            return {}
        if isinstance(self._outputs, str):
            return {self._outputs: outputs}
        if isinstance(self._outputs, dict):
            return _from_dict()
        return _from_list()

    def _validate_inputs(self, func: Callable[..., Any], inputs: Optional[Union[str, List[str], Dict[str, str]]]) -> None:
        if not inspect.isbuiltin(func):
            args, kwargs = Node._process_inputs_for_bind(inputs)
            try:
                inspect.signature(func, follow_wrapped=False).bind(*args, **kwargs)
            except Exception as exc:
                func_args = list(inspect.signature(func, follow_wrapped=False).parameters.keys())
                func_name = _get_readable_func_name(func)
                raise TypeError(f"Inputs of '{func_name}' function expected {func_args}, but got {inputs}") from exc

    def _validate_unique_outputs(self) -> None:
        cnt: Counter = Counter(self.outputs)
        diff: Set[str] = {k for k in cnt if cnt[k] > 1}
        if diff:
            raise ValueError(f'Failed to create node {self} due to duplicate output(s) {diff}.\nNode outputs must be unique.')

    def _validate_inputs_dif_than_outputs(self) -> None:
        common_in_out: Set[str] = set(map(_strip_transcoding, self.inputs)).intersection(set(map(_strip_transcoding, self.outputs)))
        if common_in_out:
            raise ValueError(f'Failed to create node {self}.\nA node cannot have the same inputs and outputs even if they are transcoded: {common_in_out}')

    @staticmethod
    def _process_inputs_for_bind(
        inputs: Optional[Union[str, List[str], Dict[str, str]]]
    ) -> Tuple[List[str], Dict[str, str]]:
        inputs_copy = copy.copy(inputs)
        args: List[str] = []
        kwargs: Dict[str, str] = {}
        if isinstance(inputs_copy, str):
            args = [inputs_copy]
        elif isinstance(inputs_copy, list):
            args = inputs_copy
        elif isinstance(inputs_copy, dict):
            kwargs = inputs_copy
        return (args, kwargs)


def _node_error_message(msg: str) -> str:
    return f'Invalid Node definition: {msg}\nFormat should be: node(function, inputs, outputs)'


def node(
    func: Callable[..., Any],
    inputs: Optional[Union[str, List[str], Dict[str, str]]] = None,
    outputs: Optional[Union[str, List[str], Dict[str, str]]] = None,
    *,
    name: Optional[str] = None,
    tags: Optional[Union[str, Iterable[str]]] = None,
    confirms: Optional[Union[str, List[str]]] = None,
    namespace: Optional[str] = None,
) -> Node:
    return Node(func, inputs, outputs, name=name, tags=tags, confirms=confirms, namespace=namespace)


def _dict_inputs_to_list(func: Callable[..., Any], inputs: Dict[str, str]) -> List[str]:
    sig = inspect.signature(func, follow_wrapped=False).bind(**inputs)
    return [*sig.args, *sig.kwargs.values()]


def _to_list(
    element: Optional[Union[str, List[str], Dict[str, str], Iterable[str]]]
) -> List[str]:
    if element is None:
        return []
    if isinstance(element, str):
        return [element]
    if isinstance(element, dict):
        return list(element.values())
    return list(element)


def _get_readable_func_name(func: Callable[..., Any]) -> str:
    if hasattr(func, '__name__'):
        return func.__name__
    name = repr(func)
    if 'functools.partial' in name:
        name = '<partial>'
    return name