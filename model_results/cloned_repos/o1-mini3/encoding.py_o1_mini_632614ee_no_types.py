import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar
from mypy_extensions import TypedDict
from monkeytype.compat import is_any, is_generic, is_union, qualname_of_generic
from monkeytype.db.base import CallTraceThunk
from monkeytype.exceptions import InvalidTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType, NotImplementedType, is_typed_dict, mappingproxy
from monkeytype.util import get_func_in_module, get_name_in_module
logger = logging.getLogger(__name__)
TypeDict = Dict[str, Any]

def typed_dict_to_dict(typ):
    elem_types: Dict[str, Any] = {}
    for k, v in typ.__annotations__.items():
        elem_types[k] = type_to_dict(v)
    return {'module': typ.__module__, 'qualname': typ.__qualname__, 'elem_types': elem_types, 'is_typed_dict': True}

def type_to_dict(typ):
    """Convert a type into a dictionary representation that we can store.

    The dictionary must:
        1. Be encodable as JSON
        2. Contain enough information to let us reify the type
    """
    if is_typed_dict(typ):
        return typed_dict_to_dict(typ)
    if is_union(typ):
        qualname = 'Union'
    elif is_any(typ):
        qualname = 'Any'
    elif is_generic(typ):
        qualname = qualname_of_generic(typ)
    else:
        qualname = typ.__qualname__
    d: TypeDict = {'module': typ.__module__, 'qualname': qualname}
    elem_types = getattr(typ, '__args__', None)
    is_bare_generic = typ in {Dict, List, Tuple}
    if not is_bare_generic and elem_types is not None and is_generic(typ):
        if elem_types == ((),):
            elem_types = ()
        d['elem_types'] = [type_to_dict(t) for t in elem_types]
    return d
_HIDDEN_BUILTIN_TYPES: Dict[str, Type[Any]] = {'NoneType': NoneType, 'NotImplementedType': NotImplementedType, 'mappingproxy': mappingproxy}

def typed_dict_from_dict(d):
    return TypedDict(d['qualname'], {k: type_from_dict(v) for k, v in d['elem_types'].items()})

def type_from_dict(d):
    """Given a dictionary produced by type_to_dict, return the equivalent type.

    Raises:
        NameLookupError if we can't reify the specified type
        InvalidTypeError if the named type isn't actually a type
    """
    module, qualname = (d['module'], d['qualname'])
    if d.get('is_typed_dict', False):
        return typed_dict_from_dict(d)
    if module == 'builtins' and qualname in _HIDDEN_BUILTIN_TYPES:
        typ = _HIDDEN_BUILTIN_TYPES[qualname]
    else:
        typ = get_name_in_module(module, qualname)
    if not (isinstance(typ, type) or is_any(typ) or is_generic(typ)):
        raise InvalidTypeError(f"Attribute specified by '{qualname}' in module '{module}' is of type {type(typ)}, not type.")
    elem_type_dicts = d.get('elem_types')
    if elem_type_dicts is not None and is_generic(typ):
        elem_types = tuple((type_from_dict(e) for e in elem_type_dicts))
        typ = typ[elem_types]
    return typ

def type_to_json(typ):
    """Encode the supplied type as json using type_to_dict."""
    type_dict = type_to_dict(typ)
    return json.dumps(type_dict, sort_keys=True)

def type_from_json(typ_json):
    """Reify a type from the format produced by type_to_json."""
    type_dict = json.loads(typ_json)
    return type_from_dict(type_dict)

def arg_types_to_json(arg_types):
    """Encode the supplied argument types as json"""
    type_dict = {name: type_to_dict(typ) for name, typ in arg_types.items()}
    return json.dumps(type_dict, sort_keys=True)

def arg_types_from_json(arg_types_json):
    """Reify the encoded argument types from the format produced by arg_types_to_json."""
    arg_types = json.loads(arg_types_json)
    return {name: type_from_dict(type_dict) for name, type_dict in arg_types.items()}
TypeEncoder = Callable[[Type[Any]], str]

def maybe_encode_type(encode, typ):
    if typ is None:
        return None
    return encode(typ)
TypeDecoder = Callable[[str], Type[Any]]

def maybe_decode_type(decode, encoded):
    if encoded is None or encoded == 'null':
        return None
    return decode(encoded)
CallTraceRowT = TypeVar('CallTraceRowT', bound='CallTraceRow')

class CallTraceRow(CallTraceThunk):
    """A semi-structured call trace where each field has been json encoded."""

    def __init__(self, module, qualname, arg_types, return_type, yield_type):
        self.module = module
        self.qualname = qualname
        self.arg_types = arg_types
        self.return_type = return_type
        self.yield_type = yield_type

    @classmethod
    def from_trace(cls, trace):
        module: str = trace.func.__module__
        qualname: str = trace.func.__qualname__
        arg_types: str = arg_types_to_json(trace.arg_types)
        return_type: Optional[str] = maybe_encode_type(type_to_json, trace.return_type)
        yield_type: Optional[str] = maybe_encode_type(type_to_json, trace.yield_type)
        return cls(module, qualname, arg_types, return_type, yield_type)

    def to_trace(self):
        function = get_func_in_module(self.module, self.qualname)
        arg_types: Dict[str, Type[Any]] = arg_types_from_json(self.arg_types)
        return_type: Optional[Type[Any]] = maybe_decode_type(type_from_json, self.return_type)
        yield_type: Optional[Type[Any]] = maybe_decode_type(type_from_json, self.yield_type)
        return CallTrace(function, arg_types, return_type, yield_type)

    def __eq__(self, other):
        if isinstance(other, CallTraceRow):
            return (self.module, self.qualname, self.arg_types, self.return_type, self.yield_type) == (other.module, other.qualname, other.arg_types, other.return_type, other.yield_type)
        return NotImplemented

def serialize_traces(traces):
    """Serialize an iterable of CallTraces to an iterable of CallTraceRow.

    Catches and logs exceptions, so a failure to serialize one CallTrace doesn't
    lose all traces.

    """
    for trace in traces:
        try:
            yield CallTraceRow.from_trace(trace)
        except Exception:
            logger.exception('Failed to serialize trace')