from __future__ import annotations
import copy
import inspect
import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Union, Generator
from warnings import warn
from more_itertools import spy, unzip
from .transcoding import _strip_transcoding
if False:  # TYPE_CHECKING
    from collections.abc import Iterable

InputsType = Optional[Union[str, List[str], Dict[str, str]]]
OutputsType = Optional[Union[str, List[str], Dict[str, str]]]
TagsType = Optional[Union[str, List[str], Set[str]]]
ConfirmsType = Optional[Union[str, List[str]]]

class Node:
    def __init__(
        self,
        func: Callable[..., Any],
        inputs: InputsType,
        outputs: OutputsType,
        *,
        name: Optional[str] = None,
        tags: TagsType = None,
        confirms: ConfirmsType = None,
        namespace: Optional[str] = None,
    ) -> None:
        if not callable(func):
            raise ValueError(_node_error_message(f"first argument must be a function, not '{type(func).__name__}'."))
        if inputs and (not isinstance(inputs, (list, dict, str))):
            raise ValueError(_node_error_message(f"'inputs' type must be one of [String, List, Dict, None], not '{type(inputs).__name__}'."))
        for _input in _to_list(inputs):
            if not isinstance(_input, str):
                raise ValueError(
                    _node_error_message(
                        f"names of variables used as inputs to the function must be of 'String' type, but {_input} from {inputs} is '{type(_input)}'."
                    )
                )
        if outputs and (not isinstance(outputs, (list, dict, str))):
            raise ValueError(_node_error_message(f"'outputs' type must be one of [String, List, Dict, None], not '{type(outputs).__name__}'."))
        for _output in _to_list(outputs):
            if not isinstance(_output, str):
                raise ValueError(
                    _node_error_message(
                        f"names of variables used as outputs of the function must be of 'String' type, but {_output} from {outputs} is '{type(_output)}'."
                    )
                )
        if not inputs and (not outputs):
            raise ValueError(_node_error_message("it must have some 'inputs' or 'outputs'."))
        self._validate_inputs(func, inputs)
        self._func: Callable[..., Any] = func
        self._inputs: InputsType = inputs
        self._outputs: OutputsType = outputs
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
        self._confirms: ConfirmsType = confirms

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
    def _unique_key(self) -> Any:
        def hashable(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted(value.items()))
            if isinstance(value, list):
                return tuple(value)
            return value
        return (self.name, hashable(self._inputs), hashable(self._outputs))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented  # type: ignore
        return self._unique_key == other._unique_key

    def __lt__(self, other: Node) -> bool:
        if not isinstance(other, Node):
            return NotImplemented  # type: ignore
        return self._unique_key < other._unique_key

    def __hash__(self) -> int:
        return hash(self._unique_key)

    def __str__(self) -> str:
        def _set_to_str(xset: List[str]) -> str:
            return f"[{';'.join(xset)}]"
        out_str: str = _set_to_str(self.outputs) if self._outputs else 'None'
        in_str: str = _set_to_str(self.inputs) if self._inputs else 'None'
        prefix: str = self._name + ': ' if self._name else ''
        return prefix + f'{self._func_name}({in_str}) -> {out_str}'

    def __repr__(self) -> str:
        return f'Node({self._func_name}, {self._inputs!r}, {self._outputs!r}, {self._name!r})'

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        return self.run(inputs=kwargs)

    @property
    def _func_name(self) -> str:
        name: str = _get_readable_func_name(self._func)
        if name == '<partial>':
            warn(
                f"The node producing outputs '{self.outputs}' is made from a 'partial' function. Partial functions do not have a '__name__' attribute: consider using 'functools.update_wrapper' for better log messages."
            )
        return name

    @property
    def func(self) -> Callable[..., Any]:
        """Exposes the underlying function of the node.

        Returns:
           Return the underlying function of the node.
        """
        return self._func

    @func.setter
    def func(self, func: Callable[..., Any]) -> None:
        self._func = func

    @property
    def tags(self) -> Set[str]:
        """Return the tags assigned to the node.

        Returns:
            Return the set of all assigned tags to the node.
        """
        return set(self._tags)

    def tag(self, tags: TagsType) -> Node:
        """Create a new ``Node`` which is an exact copy of the current one,
            but with more tags added to it.

        Args:
            tags: The tags to be added to the new node.

        Returns:
            A copy of the current ``Node`` object with the tags added.
        """
        return self._copy(tags=self.tags | set(_to_list(tags)))

    @property
    def name(self) -> str:
        """Node's name.

        Returns:
            Node's name if provided or the name of its function.
        """
        node_name: str = self._name or str(self)
        if self.namespace:
            return f'{self.namespace}.{node_name}'
        return node_name

    @property
    def short_name(self) -> str:
        """Node's name.

        Returns:
            Returns a short, user-friendly name that is not guaranteed to be unique.
            The namespace is stripped out of the node name.
        """
        if self._name:
            return self._name
        return self._func_name.replace('_', ' ').title()

    @property
    def namespace(self) -> Optional[str]:
        """Node's namespace.

        Returns:
            String representing node's namespace, typically from outer to inner scopes.
        """
        return self._namespace

    @property
    def inputs(self) -> List[str]:
        """Return node inputs as a list, in the order required to bind them properly to
        the node's function.

        Returns:
            Node input names as a list.
        """
        if isinstance(self._inputs, dict):
            return _dict_inputs_to_list(self._func, self._inputs)
        return _to_list(self._inputs)

    @property
    def outputs(self) -> List[str]:
        """Return node outputs as a list preserving the original order
            if possible.

        Returns:
            Node output names as a list.
        """
        return _to_list(self._outputs)

    @property
    def confirms(self) -> List[str]:
        """Return dataset names to confirm as a list.

        Returns:
            Dataset names to confirm as a list.
        """
        return _to_list(self._confirms)

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run this node using the provided inputs and return its results
        in a dictionary.

        Args:
            inputs: Dictionary of inputs as specified at the creation of
                the node.

        Raises:
            ValueError: In the following cases:
                a) The node function inputs are incompatible with the node
                input definition.
                Example 1: node definition input is a list of 2
                DataFrames, whereas only 1 was provided or 2 different ones
                were provided.
                b) The node function outputs are incompatible with the node
                output definition.
                Example 1: node function definition is a dictionary,
                whereas function returns a list.
                Example 2: node definition output is a list of 5 strings,
                whereas the function returns a list of 4 objects.
            Exception: Any exception thrown during execution of the node.

        Returns:
            All produced node outputs are returned in a dictionary, where the
            keys are defined by the node outputs.
        """
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
            keys: List[str] = list(self._outputs.keys())
            names: List[str] = list(self._outputs.values())
            if not isinstance(result, dict):
                raise ValueError(f'Failed to save outputs of node {self}.\nThe node output is a dictionary, whereas the function output is {type(result)}.')
            if set(keys) != set(result.keys()):
                raise ValueError(f"Failed to save outputs of node {self!s}.\nThe node's output keys {set(result.keys())} do not match with the returned output's keys {set(keys)}.")
            if iterator:
                exploded = map(lambda x: tuple((x[k] for k in keys)), iterator)
                result = unzip(exploded)
            else:
                result = tuple((result[k] for k in keys))
            return dict(zip(names, result))

        def _from_list() -> Dict[str, Any]:
            result, iterator = (outputs, None)
            if inspect.isgenerator(outputs):
                (result,), iterator = spy(outputs)
            if not isinstance(result, (list, tuple)):
                raise ValueError(f"Failed to save outputs of node {self!s}.\nThe node definition contains a list of outputs {self._outputs}, whereas the node function returned a '{type(result).__name__}'.")
            if len(result) != len(self._outputs):
                raise ValueError(f'Failed to save outputs of node {self!s}.\nThe node function returned {len(result)} output(s), whereas the node definition contains {len(self._outputs)} output(s).')
            if iterator:
                result = unzip(iterator)
            return dict(zip(self._outputs, result))
        if self._outputs is None:
            return {}
        if isinstance(self._outputs, str):
            return {self._outputs: outputs}
        if isinstance(self._outputs, dict):
            return _from_dict()
        return _from_list()

    def _validate_inputs(self, func: Callable[..., Any], inputs: InputsType) -> None:
        if not inspect.isbuiltin(func):
            args, kwargs = self._process_inputs_for_bind(inputs)
            try:
                inspect.signature(func, follow_wrapped=False).bind(*args, **kwargs)
            except Exception as exc:
                func_args = list(inspect.signature(func, follow_wrapped=False).parameters.keys())
                func_name = _get_readable_func_name(func)
                raise TypeError(f"Inputs of '{func_name}' function expected {func_args}, but got {inputs}") from exc

    def _validate_unique_outputs(self) -> None:
        cnt = Counter(self.outputs)
        diff = {k for k in cnt if cnt[k] > 1}
        if diff:
            raise ValueError(f'Failed to create node {self} due to duplicate output(s) {diff}.\nNode outputs must be unique.')

    def _validate_inputs_dif_than_outputs(self) -> None:
        common_in_out = set(map(_strip_transcoding, self.inputs)).intersection(set(map(_strip_transcoding, self.outputs)))
        if common_in_out:
            raise ValueError(f'Failed to create node {self}.\nA node cannot have the same inputs and outputs even if they are transcoded: {common_in_out}')

    @staticmethod
    def _process_inputs_for_bind(inputs: InputsType) -> (List[Any], Dict[str, Any]):
        inputs_copy = copy.copy(inputs)
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
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
    inputs: InputsType,
    outputs: OutputsType,
    *,
    name: Optional[str] = None,
    tags: TagsType = None,
    confirms: ConfirmsType = None,
    namespace: Optional[str] = None,
) -> Node:
    """Create a node in the pipeline by providing a function to be called
    along with variable names for inputs and/or outputs.

    Args:
        func: A function that corresponds to the node logic. The function
            should have at least one input or output.
        inputs: The name or the list of the names of variables used as inputs
            to the function. The number of names should match the number of
            arguments in the definition of the provided function. When
            dict[str, str] is provided, variable names will be mapped to
            function argument names.
        outputs: The name or the list of the names of variables used as outputs
            to the function. The number of names should match the number of
            outputs returned by the provided function. When dict[str, str]
            is provided, variable names will be mapped to the named outputs the
            function returns.
        name: Optional node name to be used when displaying the node in logs or
            any other visualisations.
        tags: Optional set of tags to be applied to the node.
        confirms: Optional name or the list of the names of the datasets
            that should be confirmed. This will result in calling ``confirm()``
            method of the corresponding dataset instance. Specified dataset
            names do not necessarily need to be present in the node ``inputs``
            or ``outputs``.
        namespace: Optional node namespace.

    Returns:
        A Node object with mapped inputs, outputs and function.
    """
    return Node(func, inputs, outputs, name=name, tags=tags, confirms=confirms, namespace=namespace)

def _dict_inputs_to_list(func: Callable[..., Any], inputs: Dict[str, str]) -> List[str]:
    """Convert a dict representation of the node inputs to a list, ensuring
    the appropriate order for binding them to the node's function.
    """
    sig = inspect.signature(func, follow_wrapped=False).bind(**inputs)
    return [*sig.args, *sig.kwargs.values()]

def _to_list(element: Union[str, List[str], Dict[str, str], None]) -> List[str]:
    """Make a list out of node inputs/outputs.

    Returns:
        list[str]: Node input/output names as a list to standardise.
    """
    if element is None:
        return []
    if isinstance(element, str):
        return [element]
    if isinstance(element, dict):
        return list(element.values())
    return list(element)

def _get_readable_func_name(func: Callable[..., Any]) -> str:
    """Get a user-friendly readable name of the function provided.

    Returns:
        str: readable name of the provided callable func.
    """
    if hasattr(func, '__name__'):
        return func.__name__
    name = repr(func)
    if 'functools.partial' in name:
        name = '<partial>'
    return name