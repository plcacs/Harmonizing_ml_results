#!/usr/bin/env python3
"""Helpers to help with encoding Home Assistant objects in JSON."""
from collections import deque
from collections.abc import Callable
import datetime
from functools import partial
import json
import logging
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Type, Union, Final, cast, TYPE_CHECKING
import orjson

from homeassistant.util.file import write_utf8_file, write_utf8_file_atomic
from homeassistant.util.json import (
    JSON_DECODE_EXCEPTIONS as _JSON_DECODE_EXCEPTIONS,
    JSON_ENCODE_EXCEPTIONS as _JSON_ENCODE_EXCEPTIONS,
    SerializationError,
    format_unserializable_data,
    json_loads as _json_loads,
)
from .deprecation import DeprecatedConstant, all_with_deprecated_constants, check_if_deprecated_constant, deprecated_function, dir_with_deprecated_constants

_DEPRECATED_JSON_DECODE_EXCEPTIONS: Final = DeprecatedConstant(_JSON_DECODE_EXCEPTIONS, 'homeassistant.util.json.JSON_DECODE_EXCEPTIONS', '2025.8')
_DEPRECATED_JSON_ENCODE_EXCEPTIONS: Final = DeprecatedConstant(_JSON_ENCODE_EXCEPTIONS, 'homeassistant.util.json.JSON_ENCODE_EXCEPTIONS', '2025.8')
json_loads: Callable[[str], Any] = deprecated_function('homeassistant.util.json.json_loads', breaks_in_ha_version='2025.8')(_json_loads)
__getattr__ = partial(check_if_deprecated_constant, module_globals=globals())
__dir__ = partial(dir_with_deprecated_constants, module_globals_keys=[*globals().keys()])
__all__ = all_with_deprecated_constants(globals())
_LOGGER: Final = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    """JSONEncoder that supports Home Assistant objects."""

    def default(self, o: Any) -> Any:
        """Convert Home Assistant objects.

        Hand other objects to the original method.
        """
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        if isinstance(o, set):
            return list(o)
        if hasattr(o, 'as_dict'):
            return o.as_dict()
        return json.JSONEncoder.default(self, o)


def json_encoder_default(obj: Any) -> Any:
    """Convert Home Assistant objects.

    Hand other objects to the original method.
    """
    if hasattr(obj, 'json_fragment'):
        return obj.json_fragment
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, float):
        return float(obj)
    if hasattr(obj, 'as_dict'):
        return obj.as_dict()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if TYPE_CHECKING:
    def json_bytes(obj: Any) -> bytes:
        """Dump json bytes."""
        ...
else:
    json_bytes: Callable[[Any], bytes] = partial(orjson.dumps, option=orjson.OPT_NON_STR_KEYS, default=json_encoder_default)
    "Dump json bytes."


class ExtendedJSONEncoder(JSONEncoder):
    """JSONEncoder that supports Home Assistant objects and falls back to repr(o)."""

    def default(self, o: Any) -> Any:
        """Convert certain objects.

        Fall back to repr(o).
        """
        if isinstance(o, datetime.timedelta):
            return {'__type': str(type(o)), 'total_seconds': o.total_seconds()}
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        if isinstance(o, (datetime.date, datetime.time)):
            return {'__type': str(type(o)), 'isoformat': o.isoformat()}
        try:
            return super().default(o)
        except TypeError:
            return {'__type': str(type(o)), 'repr': repr(o)}


def _strip_null(obj: Any) -> Any:
    """Strip NUL from an object."""
    if isinstance(obj, str):
        return obj.split('\x00', 1)[0]
    if isinstance(obj, dict):
        return {key: _strip_null(o) for key, o in obj.items()}
    if isinstance(obj, list):
        return [_strip_null(o) for o in obj]
    return obj


def json_bytes_strip_null(data: Any) -> bytes:
    """Dump json bytes after terminating strings at the first NUL."""
    result: bytes = json_bytes(data)
    if b'\\u0000' not in result:
        return result
    return json_bytes(_strip_null(orjson.loads(result)))


json_fragment: Final = orjson.Fragment


def json_dumps(data: Any) -> str:
    """Dump json string.

    orjson supports serializing dataclasses natively which
    eliminates the need to implement as_dict in many places
    when the data is already in a dataclass. This works
    well as long as all the data in the dataclass can also
    be serialized.

    If it turns out to be a problem we can disable this
    with option |= orjson.OPT_PASSTHROUGH_DATACLASS and it
    will fallback to as_dict
    """
    return json_bytes(data).decode('utf-8')


json_bytes_sorted: Callable[[Any], bytes] = partial(
    orjson.dumps,
    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS,
    default=json_encoder_default
)
'Dump json bytes with keys sorted.'


def json_dumps_sorted(data: Any) -> str:
    """Dump json string with keys sorted."""
    return json_bytes_sorted(data).decode('utf-8')


JSON_DUMP: Callable[[Any], str] = json_dumps


def _orjson_default_encoder(data: Any) -> str:
    """JSON encoder that uses orjson with hass defaults and returns a str."""
    return _orjson_bytes_default_encoder(data).decode('utf-8')


def _orjson_bytes_default_encoder(data: Any) -> bytes:
    """JSON encoder that uses orjson with hass defaults and returns bytes."""
    return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS, default=json_encoder_default)


def save_json(
    filename: Union[str, Path],
    data: Any,
    private: bool = False,
    *,
    encoder: Optional[Type[json.JSONEncoder]] = None,
    atomic_writes: bool = False,
) -> None:
    """Save JSON data to a file."""
    try:
        if encoder and encoder is not JSONEncoder:
            mode: str = 'w'
            dump: Callable[[Any], str] = json.dumps
            json_data: str = json.dumps(data, indent=2, cls=encoder)
        else:
            mode = 'wb'
            dump = _orjson_default_encoder
            json_data = _orjson_bytes_default_encoder(data)
    except TypeError as error:
        formatted_data: Any = format_unserializable_data(find_paths_unserializable_data(data, dump=dump))
        msg: str = f'Failed to serialize to JSON: {filename}. Bad data at {formatted_data}'
        _LOGGER.error(msg)
        raise SerializationError(msg) from error
    method: Callable[[Union[str, Path], Any, bool], None] = write_utf8_file_atomic if atomic_writes else write_utf8_file
    method(filename, json_data, private, mode=mode)


def find_paths_unserializable_data(
    bad_data: Any,
    *,
    dump: Callable[[Any], Any] = json.dumps,
) -> Dict[str, Any]:
    """Find the paths to unserializable data.

    This method is slow! Only use for error handling.
    """
    from homeassistant.core import Event, State

    to_process: Deque[tuple[Any, str]] = deque([(bad_data, '$')])
    invalid: Dict[str, Any] = {}
    while to_process:
        obj, obj_path = to_process.popleft()
        try:
            dump(obj)
            continue
        except (ValueError, TypeError):
            pass
        if hasattr(obj, 'as_dict'):
            desc: str = obj.__class__.__name__
            if isinstance(obj, State):
                desc += f': {obj.entity_id}'
            elif isinstance(obj, Event):
                desc += f': {obj.event_type}'
            obj_path += f'({desc})'
            obj = obj.as_dict()
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    dump({key: None})
                except TypeError:
                    invalid[f'{obj_path}<key: {key}>'] = key
                else:
                    to_process.append((value, f'{obj_path}.{key}'))
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                to_process.append((value, f'{obj_path}[{idx}]'))
        else:
            invalid[obj_path] = obj
    return invalid
