import copy
import decimal
import logging
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
import numpy as np
import pandas as pd
import simplejson
from flask_babel.speaklater import LazyString
from jsonpath_ng import parse
from simplejson import JSONDecodeError
from superset.constants import PASSWORD_MASK
from superset.utils.dates import datetime_to_epoch, EPOCH

logging.getLogger('MARKDOWN').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class DashboardEncoder(simplejson.JSONEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sort_keys: bool = True

    def default(self, o: Any) -> Any:
        if isinstance(o, uuid.UUID):
            return str(o)
        try:
            vals = {k: v for k, v in o.__dict__.items() if k != '_sa_instance_state'}
            return {f'__{o.__class__.__name__}__': vals}
        except Exception:
            if isinstance(o, datetime):
                return {'__datetime__': o.replace(microsecond=0).isoformat()}
            return simplejson.JSONEncoder(sort_keys=True).default(o)


def format_timedelta(time_delta: timedelta) -> str:
    """
    Ensures negative time deltas are easily interpreted by humans
    """
    if time_delta < timedelta(0):
        return '-' + str(abs(time_delta))
    return str(time_delta)


def base_json_conv(obj: Any) -> Any:
    """
    Tries to convert additional types to JSON compatible forms.
    """
    if isinstance(obj, memoryview):
        obj = obj.tobytes()
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (uuid.UUID, time, LazyString)):
        return str(obj)
    if isinstance(obj, timedelta):
        return format_timedelta(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            try:
                return obj.decode('utf-16')
            except Exception:
                return '[bytes]'
    raise TypeError(f'Unserializable object {obj} of type {type(obj)}')


def json_iso_dttm_ser(obj: Any, pessimistic: bool = False) -> Any:
    """
    A JSON serializer that deals with dates by serializing them to ISO 8601.
    """
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    try:
        return base_json_conv(obj)
    except TypeError:
        if pessimistic:
            logger.error('Failed to serialize %s', obj)
            return f'Unserializable [{type(obj)}]'
        raise


def pessimistic_json_iso_dttm_ser(obj: Any) -> Any:
    """Proxy to call json_iso_dttm_ser in a pessimistic way"""
    return json_iso_dttm_ser(obj, pessimistic=True)


def json_int_dttm_ser(obj: Any) -> Any:
    """
    A JSON serializer that deals with dates by serializing them to EPOCH.
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return datetime_to_epoch(obj)
    if isinstance(obj, date):
        return (obj - EPOCH.date()).total_seconds() * 1000
    return base_json_conv(obj)


def json_dumps_w_dates(payload: Any, sort_keys: bool = False) -> str:
    """Dumps payload to JSON with Datetime objects properly converted"""
    return dumps(payload, default=json_int_dttm_ser, sort_keys=sort_keys)


def validate_json(obj: Union[str, bytes, bytearray, None]) -> None:
    """
    A JSON Validator that validates an object of bytes, bytes array or string
    to be in valid JSON format.
    """
    if obj:
        try:
            loads(obj)
        except JSONDecodeError as ex:
            logger.error('JSON is not valid %s', str(ex), exc_info=True)
            raise


def dumps(
    obj: Any,
    default: Callable[[Any], Any] = json_iso_dttm_ser,
    allow_nan: bool = False,
    ignore_nan: bool = True,
    sort_keys: bool = False,
    indent: Optional[int] = None,
    separators: Optional[Tuple[Any, Any]] = None,
    cls: Optional[Type[simplejson.JSONEncoder]] = None,
    encoding: Optional[str] = 'utf-8'
) -> str:
    """
    Dumps object to compatible JSON format.
    """
    results_string = ''
    dumps_kwargs: Dict[str, Any] = {
        'default': default,
        'allow_nan': allow_nan,
        'ignore_nan': ignore_nan,
        'sort_keys': sort_keys,
        'indent': indent,
        'separators': separators,
        'cls': cls,
        'encoding': encoding,
    }
    try:
        results_string = simplejson.dumps(obj, **dumps_kwargs)
    except UnicodeDecodeError:
        dumps_kwargs['encoding'] = None
        results_string = simplejson.dumps(obj, **dumps_kwargs)
    return results_string


def loads(
    obj: Union[str, bytes],
    encoding: Optional[str] = None,
    allow_nan: bool = False,
    object_hook: Optional[Callable[[Dict[str, Any]], Any]] = None
) -> Any:
    """
    Deserialize instance to a Python object.
    """
    return simplejson.loads(obj, encoding=encoding, allow_nan=allow_nan, object_hook=object_hook)


def redact_sensitive(payload: Any, sensitive_fields: List[str]) -> Any:
    """
    Redacts sensitive fields from a payload.
    """
    redacted_payload = copy.deepcopy(payload)
    for json_path in sensitive_fields:
        jsonpath_expr = parse(json_path)
        for match in jsonpath_expr.find(redacted_payload):
            match.context.value[match.path.fields[0]] = PASSWORD_MASK
    return redacted_payload


def reveal_sensitive(old_payload: Any, new_payload: Any, sensitive_fields: List[str]) -> Any:
    """
    Reveals sensitive fields from a payload when not modified.
    """
    revealed_payload = copy.deepcopy(new_payload)
    for json_path in sensitive_fields:
        jsonpath_expr = parse(json_path)
        for match in jsonpath_expr.find(revealed_payload):
            if match.value == PASSWORD_MASK:
                old_value = match.full_path.find(old_payload)
                match.context.value[match.path.fields[0]] = old_value[0].value
    return revealed_payload
