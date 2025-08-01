import copy
from itertools import chain
import json
import logging
import os
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping
from os import PathLike
from typing import Any, Dict, List, Union, Optional, TypeVar, Iterable, Set, Tuple

try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:

    def evaluate_file(filename: Union[str, PathLike], **_kwargs: Any) -> str:
        logger.warning(f'error loading _jsonnet (this is expected on Windows), treating {filename} as plain json')
        with open(filename, 'r') as evaluation_file:
            return evaluation_file.read()

    def evaluate_snippet(_filename: str, expr: str, **_kwargs: Any) -> str:
        logger.warning('error loading _jsonnet (this is expected on Windows), treating snippet as plain json')
        return expr

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
logger = logging.getLogger(__name__)

T = TypeVar('T', dict, list)

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

def with_overrides(original: Union[Dict[str, Any], List[Any]], overrides_dict: Dict[str, Any], prefix: str = '') -> Union[Dict[str, Any], List[Any]]:
    if isinstance(original, list):
        merged: Union[List[Any], Dict[str, Any]] = [None] * len(original)
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
        key_str = str(key)
        if key_str in overrides_dict:
            merged[key] = copy.deepcopy(overrides_dict[key_str])
            used_override_keys.add(key_str)
        else:
            overrides_subdict: Dict[str, Any] = {}
            for o_key in overrides_dict:
                if o_key.startswith(f'{key_str}.'):
                    overrides_subdict[o_key[len(f'{key_str}.'):]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                if isinstance(original, list):
                    if not isinstance(original[key], (dict, list)):
                        raise ConfigurationError(f"Cannot apply nested overrides to non-dict/list original value at {prefix}{key_str}")
                merged[key] = with_overrides(original[key], overrides_subdict, prefix=prefix + f'{key_str}.')
            else:
                merged[key] = copy.deepcopy(original[key])
    unused_override_keys = [prefix + key for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f'overrides dict contains unused keys: {unused_override_keys}')
    return merged

def parse_overrides(serialized_overrides: Optional[str], ext_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if serialized_overrides:
        ext_vars_combined = {**_environment_variables(), **(ext_vars or {})}
        return json.loads(evaluate_snippet('', serialized_overrides, ext_vars=ext_vars_combined))
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

    There are currently two benefits of a `Params` object over a plain dictionary for parameter
    passing:

    1. We handle a few kinds of parameter validation, including making sure that parameters
       representing discrete choices actually have acceptable values, and making sure no extra
       parameters are passed.
    2. We log all parameter reads, including default values.  This gives a more complete
       specification of the actual parameters used than is given in a JSON file, because
       those may not specify what default values were used, whereas this will log them.

    !!! Consumption
        The convention for using a `Params` object in AllenNLP is that you will consume the parameters
        as you read them, so that there are none left when you've read everything you expect.  This
        lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
        that the parameter dictionary is empty.  You should do this when you're done handling
        parameters, by calling `Params.assert_empty`.
    """
    DEFAULT = object()

    def __init__(self, params: Union[Dict[str, Any], List[Any]], history: str = '') -> None:
        self.params: Union[Dict[str, Any], List[Any]] = _replace_none(params)
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
            raise ValueError('Cannot convert variable to bool: ' + value)

    def get(self, key: str, default: Any = DEFAULT) -> Any:
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        default_value = None if default is self.DEFAULT else default
        value = self.params.get(key, default_value)
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
        the given choices. Note that this `pops` the key from params, modifying the dictionary,
        consistent with how parameters are processed in this codebase.

        # Parameters

        key: `str`

            Key to get the value from in the param dictionary

        choices: `List[Any]`

            A list of valid options for values corresponding to `key`.  For example, if you're
            specifying the type of encoder to use for some part of your model, the choices might be
            the list of encoder classes we know about and can instantiate.  If the value we find in
            the param dictionary is not in `choices`, we raise a `ConfigurationError`, because
            the user specified an invalid value in their parameter file.

        default_to_first_choice: `bool`, optional (default = `False`)

            If this is `True`, we allow the `key` to not be present in the parameter
            dictionary.  If the key is not present, we will use the return as the value the first
            choice in the `choices` list.  If this is `False`, we raise a
            `ConfigurationError`, because specifying the `key` is required (e.g., you `have` to
            specify your model class when running an experiment, but you can feel free to use
            default settings for encoders if you want).

        allow_class_names: `bool`, optional (default = `True`)

            If this is `True`, then we allow unknown choices that look like fully-qualified class names.
            This is to allow e.g. specifying a model type as my_library.my_model.MyModel
            and importing it on the fly. Our check for "looks like" is extremely lenient
            and consists of checking that the value contains a '.'.
        """
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        ok_because_class_name = allow_class_names and isinstance(value, str) and '.' in value
        if value not in choices and not ok_because_class_name:
            key_str = self.history + key
            message = (
                f'{value} not in acceptable choices for {key_str}: {choices}. '
                'You should either use the --include-package flag to make sure the correct module is loaded, '
                'or use a fully qualified class name in your config file like {"model": "my_module.models.MyModel"} '
                'to have it imported automatically.'
            )
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet: bool = False, infer_type_and_cast: bool = False) -> Union[Dict[str, Any], List[Any]]:
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to PyTorch code.

        # Parameters

        quiet: `bool`, optional (default = `False`)

            Whether to log the parameters before returning them as a dict.

        infer_type_and_cast: `bool`, optional (default = `False`)

            If True, we infer types and cast (e.g. things that look like floats to floats).
        """
        if infer_type_and_cast:
            params_as_dict: Union[Dict[str, Any], List[Any]] = infer_and_cast(self.params)
        else:
            params_as_dict = self.params
        if quiet:
            return params_as_dict

        def log_recursively(parameters: Union[Dict[str, Any], List[Any]], history: str) -> None:
            if isinstance(parameters, dict):
                for key, value in parameters.items():
                    if isinstance(value, dict):
                        new_local_history = history + key + '.'
                        log_recursively(value, new_local_history)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            item_history = f"{history}{key}.{i}."
                            log_recursively(item, item_history)
                    else:
                        logger.info(f'{history}{key} = {value}')
            elif isinstance(parameters, list):
                for i, item in enumerate(parameters):
                    item_history = f"{history}{i}."
                    log_recursively(item, item_history)

        log_recursively(self.params, self.history)
        return params_as_dict

    def as_flat_dict(self) -> Dict[str, Any]:
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params: Dict[str, Any] = {}

        def recurse(parameters: Union[Dict[str, Any], List[Any]], path: List[str]) -> None:
            if isinstance(parameters, dict):
                for key, value in parameters.items():
                    newpath = path + [key]
                    if isinstance(value, dict):
                        recurse(value, newpath)
                    else:
                        flat_params['.'.join(newpath)] = value
            elif isinstance(parameters, list):
                for index, value in enumerate(parameters):
                    newpath = path + [str(index)]
                    if isinstance(value, dict):
                        recurse(value, newpath)
                    else:
                        flat_params['.'.join(newpath)] = value

        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> 'Params':
        """
        Uses `copy.deepcopy()` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return copy.deepcopy(self)

    def assert_empty(self, class_name: str) -> None:
        """
        Raises a `ConfigurationError` if `self.params` is not empty.  We take `class_name` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  `class_name` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.params:
            raise ConfigurationError(f'Extra parameters passed to {class_name}: {self.params}')

    def __getitem__(self, key: str) -> Any:
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError(str(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self.params[key] = value

    def __delitem__(self, key: str) -> None:
        del self.params[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def _check_is_dict(self, new_history: str, value: Any) -> Any:
        if isinstance(value, dict):
            new_history_combined = self.history + new_history + '.'
            return Params(value, history=new_history_combined)
        if isinstance(value, list):
            return [self._check_is_dict(f'{new_history}.{i}', v) for i, v in enumerate(value)]
        return value

    @classmethod
    def from_file(cls, params_file: str, params_overrides: Union[str, Dict[str, Any]] = '', ext_vars: Optional[Dict[str, Any]] = None) -> 'Params':
        """
        Load a `Params` object from a configuration file.

        # Parameters

        params_file: `str`

            The path to the configuration file to load.

        params_overrides: `Union[str, Dict[str, Any]]`, optional (default = `""`)

            A dict of overrides that can be applied to final object.
            e.g. `{"model.embedding_dim": 10}` will change the value of "embedding_dim"
            within the "model" object of the config to 10. If you wanted to override the entire
            "model" object of the config, you could do `{"model": {"type": "other_type", ...}}`.

        ext_vars: `dict`, optional

            Our config files are Jsonnet, which allows specifying external variables
            for later substitution. Typically we substitute these using environment
            variables; however, you can also specify them here, in which case they
            take priority over environment variables.
            e.g. {"HOME_DIR": "/Users/allennlp/home"}
        """
        if ext_vars is None:
            ext_vars = {}
        cached_params_file: str = cached_path(params_file)
        ext_vars_combined: Dict[str, Any] = {**_environment_variables(), **ext_vars}
        file_dict: Union[Dict[str, Any], List[Any]] = json.loads(evaluate_file(cached_params_file, ext_vars=ext_vars_combined))
        if isinstance(params_overrides, dict):
            params_overrides_str: str = json.dumps(params_overrides)
        else:
            params_overrides_str = params_overrides
        overrides_dict: Dict[str, Any] = parse_overrides(params_overrides_str, ext_vars=ext_vars_combined)
        if overrides_dict:
            param_dict: Union[Dict[str, Any], List[Any]] = with_overrides(file_dict, overrides_dict)
        else:
            param_dict = file_dict
        return cls(param_dict)

    def to_file(self, params_file: str, preference_orders: Optional[List[List[str]]] = None) -> None:
        with open(params_file, 'w') as handle:
            json.dump(self.as_ordered_dict(preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: Optional[List[List[str]]] = None) -> OrderedDict:
        """
        Returns Ordered Dict of Params from list of partial order preferences.

        # Parameters

        preference_orders: `List[List[str]]`, optional

            `preference_orders` is list of partial preference orders. ["A", "B", "C"] means
            "A" > "B" > "C". For multiple preference_orders first will be considered first.
            Keys not found, will have last but alphabetical preference. Default Preferences:
            `[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",
            "test_data_path", "trainer", "vocabulary"], ["type"]]`
        """
        params_dict: Union[Dict[str, Any], List[Any]] = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(['dataset_reader', 'iterator', 'model', 'train_data_path', 'validation_data_path', 'test_data_path', 'trainer', 'vocabulary'])
            preference_orders.append(['type'])

        def order_func(key: str) -> Tuple:
            order_tuple = tuple(order.index(key) if key in order else len(order) for order in preference_orders)
            return order_tuple + (key,)

        def order_dict(dictionary: Union[Dict[str, Any], List[Any]], order_func_inner: callable) -> OrderedDict:
            result = OrderedDict()
            if isinstance(dictionary, dict):
                sorted_items = sorted(dictionary.items(), key=lambda item: order_func_inner(item[0]))
                for key, val in sorted_items:
                    result[key] = order_dict(val, order_func_inner) if isinstance(val, dict) else val
            elif isinstance(dictionary, list):
                for idx, item in enumerate(dictionary):
                    if isinstance(item, dict):
                        result[idx] = order_dict(item, order_func_inner)
                    else:
                        result[idx] = item
            return result

        return order_dict(params_dict, order_func)

    def get_hash(self) -> str:
        """
        Returns a hash code representing the current state of this `Params` object.  We don't
        want to implement `__hash__` because that has deeper python implications (and this is a
        mutable object), but this will give you a representation of the current state.
        We use `zlib.adler32` instead of Python's builtin `hash` because the random seed for the
        latter is reset on each new program invocation, as discussed here:
        https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3.
        """
        dumped: str = json.dumps(self.params, sort_keys=True)
        hashed: int = zlib.adler32(dumped.encode())
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
    places that the Params object is not welcome, such as inside Keras layers.  See the docstring
    of that method for more detail on how this function works.

    This method adds a `history` parameter, in the off-chance that you know it, so that we can
    reproduce `Params.pop_choice` exactly.  We default to using "?." if you don't know the
    history, so you'll have to fix that in the log if you want to actually recover the logged
    parameters.
    """
    value = Params(params, history).pop_choice(key, choices, default_to_first_choice, allow_class_names=allow_class_names)
    return value

def _replace_none(params: Any) -> Any:
    if params == 'None':
        return None
    elif isinstance(params, dict):
        return {key: _replace_none(value) for key, value in params.items()}
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params

def remove_keys_from_params(params: Union[Params, Dict[str, Any], List[Any]], keys: List[str] = ['pretrained_file', 'initializer']) -> None:
    if isinstance(params, Params):
        param_keys = list(params.keys())
        for key in keys:
            if key in param_keys:
                del params[key]
        for value in params.values():
            if isinstance(value, Params):
                remove_keys_from_params(value, keys)
    elif isinstance(params, dict):
        for key in keys:
            params.pop(key, None)
        for value in params.values():
            if isinstance(value, (dict, list)):
                remove_keys_from_params(value, keys)
    elif isinstance(params, list):
        for item in params:
            if isinstance(item, (dict, list)):
                remove_keys_from_params(item, keys)
