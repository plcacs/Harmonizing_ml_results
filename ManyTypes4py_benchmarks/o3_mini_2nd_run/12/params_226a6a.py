import copy
from itertools import chain
import json
import logging
import os
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping, Iterator
from os import PathLike
from typing import Any, Dict, List, Union, Optional, TypeVar, Iterable, Set

try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:

    def evaluate_file(filename: str, **_kwargs: Any) -> str:
        logger.warning(f'error loading _jsonnet (this is expected on Windows), treating {filename} as plain json')
        with open(filename, 'r') as evaluation_file:
            return evaluation_file.read()

    def evaluate_snippet(_filename: str, expr: str, **_kwargs: Any) -> str:
        logger.warning('error loading _jsonnet (this is expected on Windows), treating snippet as plain json')
        return expr

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)

def infer_and_cast(value: Any) -> Any:
    """
    In some cases we'll be feeding params dicts to functions we don't own;
    for example, PyTorch optimizers. In that case we can't use `pop_int`
    or similar to force casts (which means you can't specify `int` parameters
    using environment variables). This function takes something that looks JSON-like
    and recursively casts things that look like (bool, int, float) to (bool, int, float).
    """
    if isinstance(value, (int, float, bool)):
        return value
    elif isinstance(value, list):
        return [infer_and_cast(item) for item in value]
    elif isinstance(value, dict):
        return {key: infer_and_cast(item) for key, item in value.items()}
    elif isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                return value
    else:
        raise ValueError(f'cannot infer type of {value}')

def _is_encodable(value: str) -> bool:
    """
    We need to filter out environment variables that can't
    be unicode-encoded to avoid a "surrogates not allowed"
    error in jsonnet.
    """
    return value == '' or value.encode('utf-8', 'ignore') != b''

def _environment_variables() -> Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}

T = TypeVar('T', dict, list)

def with_overrides(
    original: Union[Dict[Any, Any], List[Any]],
    overrides_dict: Dict[str, Any],
    prefix: str = ''
) -> Union[Dict[Any, Any], List[Any]]:
    if isinstance(original, list):
        merged: List[Any] = [None] * len(original)
        keys = range(len(original))
    elif isinstance(original, dict):
        merged = {}
        keys = chain(original.keys(), (k for k in overrides_dict if '.' not in k and k not in original))
    elif prefix:
        raise ValueError(f"overrides for '{prefix[:-1]}.*' expected list or dict in original, found {type(original)} instead")
    else:
        raise ValueError(f'expected list or dict, found {type(original)} instead')
    used_override_keys: Set[str] = set()
    for key in keys:
        if str(key) in overrides_dict:
            merged[key] = copy.deepcopy(overrides_dict[str(key)])
            used_override_keys.add(str(key))
        else:
            overrides_subdict: Dict[str, Any] = {}
            for o_key in overrides_dict:
                if o_key.startswith(f'{key}.'):
                    overrides_subdict[o_key[len(f"{key}."):]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                merged[key] = with_overrides(original[key], overrides_subdict, prefix=prefix + f'{key}.')
            else:
                merged[key] = copy.deepcopy(original[key])
    unused_override_keys = [prefix + key for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f'overrides dict contains unused keys: {unused_override_keys}')
    return merged

def parse_overrides(serialized_overrides: str, ext_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if serialized_overrides:
        ext_vars = {**_environment_variables(), **(ext_vars or {})}
        return json.loads(evaluate_snippet('', serialized_overrides, ext_vars=ext_vars))
    else:
        return {}

def _is_dict_free(obj: Any) -> bool:
    """
    Returns False if obj is a dict, or if it's a list with an element that _has_dict.
    """
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, list):
        return all((_is_dict_free(item) for item in obj))
    else:
        return True

class Params(MutableMapping):
    """
    Represents a parameter dictionary with a history, and contains other functionality around
    parameter passing and validation for AllenNLP.
    """
    DEFAULT = object()

    def __init__(self, params: Any, history: str = '') -> None:
        self.params: Any = _replace_none(params)
        self.history: str = history

    def pop(self, key: str, default: Any = DEFAULT, keep_as_dict: bool = False) -> Any:
        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history
        (unless keep_as_dict is True, in which case we leave them as dictionaries).

        If `key` is not present in the dictionary, and no default was specified, we raise a
        `ConfigurationError`, instead of the typical `KeyError`.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                msg = f'key "{key}" is required'
                if self.history:
                    msg += f' at location "{self.history}"'
                raise ConfigurationError(msg)
        else:
            value = self.params.pop(key, default)
        if keep_as_dict or _is_dict_free(value):
            logger.info(f'{self.history}{key} = {value}')
            return value
        else:
            return self._check_is_dict(key, value)

    def pop_int(self, key: str, default: Any = DEFAULT) -> Optional[int]:
        """
        Performs a pop and coerces to an int.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: str, default: Any = DEFAULT) -> Optional[float]:
        """
        Performs a pop and coerces to a float.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: str, default: Any = DEFAULT) -> Optional[bool]:
        """
        Performs a pop and coerces to a bool.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise ValueError('Cannot convert variable to bool: ' + str(value))

    def get(self, key: str, default: Any = DEFAULT) -> Any:
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        default = None if default is self.DEFAULT else default
        value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(
        self,
        key: str,
        choices: List[Any],
        default_to_first_choice: bool = False,
        allow_class_names: bool = True
    ) -> Any:
        """
        Gets the value of `key` in the `params` dictionary, ensuring that the value is one of
        the given choices.
        """
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        ok_because_class_name = allow_class_names and '.' in value
        if value not in choices and (not ok_because_class_name):
            key_str = self.history + key
            message = f'{value} not in acceptable choices for {key_str}: {choices}. You should either use the --include-package flag to make sure the correct module is loaded, or use a fully qualified class name in your config file like {{"model": "my_module.models.MyModel"}} to have it imported automatically.'
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet: bool = False, infer_type_and_cast: bool = False) -> Dict[str, Any]:
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to PyTorch code.
        """
        if infer_type_and_cast:
            params_as_dict = infer_and_cast(self.params)
        else:
            params_as_dict = self.params
        if quiet:
            return params_as_dict

        def log_recursively(parameters: Dict[str, Any], history: str) -> None:
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + '.'
                    log_recursively(value, new_local_history)
                else:
                    logger.info(f'{history}{key} = {value}')
        log_recursively(self.params, self.history)
        return params_as_dict

    def as_flat_dict(self) -> Dict[str, Any]:
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params: Dict[str, Any] = {}

        def recurse(parameters: Dict[str, Any], path: List[str]) -> None:
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value
        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> "Params":
        """
        Uses `copy.deepcopy()` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return copy.deepcopy(self)

    def assert_empty(self, class_name: str) -> None:
        """
        Raises a `ConfigurationError` if `self.params` is not empty.
        """
        if self.params:
            raise ConfigurationError('Extra parameters passed to {}: {}'.format(class_name, self.params))

    def __getitem__(self, key: str) -> Any:
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError(str(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self.params[key] = value

    def __delitem__(self, key: str) -> None:
        del self.params[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def _check_is_dict(self, new_history: str, value: Any) -> Any:
        if isinstance(value, dict):
            new_history_combined = self.history + new_history + '.'
            return Params(value, history=new_history_combined)
        if isinstance(value, list):
            value = [self._check_is_dict(f'{new_history}.{i}', v) for i, v in enumerate(value)]
        return value

    @classmethod
    def from_file(
        cls, 
        params_file: Union[str, PathLike], 
        params_overrides: Union[str, Dict[str, Any]] = '', 
        ext_vars: Optional[Dict[str, str]] = None
    ) -> "Params":
        """
        Load a `Params` object from a configuration file.
        """
        if ext_vars is None:
            ext_vars = {}
        params_file_str = cached_path(params_file)  # type: str
        ext_vars = {**_environment_variables(), **ext_vars}
        file_dict = json.loads(evaluate_file(params_file_str, ext_vars=ext_vars))
        if isinstance(params_overrides, dict):
            params_overrides = json.dumps(params_overrides)
        overrides_dict = parse_overrides(params_overrides, ext_vars=ext_vars)
        if overrides_dict:
            param_dict = with_overrides(file_dict, overrides_dict)
        else:
            param_dict = file_dict
        return cls(param_dict)

    def to_file(self, params_file: Union[str, PathLike], preference_orders: Optional[List[List[str]]] = None) -> None:
        with open(params_file, 'w') as handle:
            json.dump(self.as_ordered_dict(preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: Optional[List[List[str]]] = None) -> OrderedDict:
        """
        Returns Ordered Dict of Params from list of partial order preferences.
        """
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(['dataset_reader', 'iterator', 'model', 'train_data_path', 'validation_data_path', 'test_data_path', 'trainer', 'vocabulary'])
            preference_orders.append(['type'])

        def order_func(key: str) -> List[Union[int, str]]:
            order_tuple = [order.index(key) if key in order else len(order) for order in preference_orders]
            return order_tuple + [key]

        def order_dict(dictionary: Dict[str, Any], order_func: Any) -> OrderedDict:
            result: OrderedDict = OrderedDict()
            for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(val, order_func) if isinstance(val, dict) else val
            return result
        return order_dict(params_dict, order_func)

    def get_hash(self) -> str:
        """
        Returns a hash code representing the current state of this `Params` object.
        """
        dumped = json.dumps(self.params, sort_keys=True)
        hashed = zlib.adler32(dumped.encode())
        return str(hashed)

    def __str__(self) -> str:
        return f'{self.history}Params({self.params})'

def pop_choice(
    params: Dict[str, Any],
    key: str,
    choices: List[Any],
    default_to_first_choice: bool = False,
    history: str = '?.',
    allow_class_names: bool = True
) -> Any:
    """
    Performs the same function as `Params.pop_choice`, but is required in order to deal with
    places that the Params object is not welcome, such as inside Keras layers.
    """
    value = Params(params, history).pop_choice(key, choices, default_to_first_choice, allow_class_names=allow_class_names)
    return value

def _replace_none(params: Any) -> Any:
    if params == 'None':
        return None
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = _replace_none(value)
        return params
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params

def remove_keys_from_params(params: Any, keys: List[str] = ['pretrained_file', 'initializer']) -> None:
    if isinstance(params, Params):
        param_keys = list(params.keys())
        for key in keys:
            if key in param_keys:
                del params[key]
        for value in params.values():
            if isinstance(value, Params):
                remove_keys_from_params(value, keys)