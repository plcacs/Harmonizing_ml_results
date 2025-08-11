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

def typed_dict_to_dict(typ: typing.Type) -> dict[typing.Text, typing.Union[dict,bool]]:
    elem_types = {}
    for k, v in typ.__annotations__.items():
        elem_types[k] = type_to_dict(v)
    return {'module': typ.__module__, 'qualname': typ.__qualname__, 'elem_types': elem_types, 'is_typed_dict': True}

def type_to_dict(typ: typing.Type) -> Union[dict, str, dict[typing.Text, typing.Union[str,tuple[typing.Literal],typing.Type,list]]]:
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
    d = {'module': typ.__module__, 'qualname': qualname}
    elem_types = getattr(typ, '__args__', None)
    is_bare_generic = typ in {Dict, List, Tuple}
    if not is_bare_generic and elem_types is not None and is_generic(typ):
        if elem_types == ((),):
            elem_types = ()
        d['elem_types'] = [type_to_dict(t) for t in elem_types]
    return d
_HIDDEN_BUILTIN_TYPES = {'NoneType': NoneType, 'NotImplementedType': NotImplementedType, 'mappingproxy': mappingproxy}

def typed_dict_from_dict(d: Union[dict[str, typing.Any], dict[str, str], dict]) -> TypedDict:
    return TypedDict(d['qualname'], {k: type_from_dict(v) for k, v in d['elem_types'].items()})

def type_from_dict(d: Union[dict, dict[str, typing.Any]]) -> Union[dict, dict[str, T], str, typing.Type]:
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

def type_to_json(typ: typing.Type):
    """Encode the supplied type as json using type_to_dict."""
    type_dict = type_to_dict(typ)
    return json.dumps(type_dict, sort_keys=True)

def type_from_json(typ_json: Union[str, int, Exception]):
    """Reify a type from the format produced by type_to_json."""
    type_dict = json.loads(typ_json)
    return type_from_dict(type_dict)

def arg_types_to_json(arg_types: Any):
    """Encode the supplied argument types as json"""
    type_dict = {name: type_to_dict(typ) for name, typ in arg_types.items()}
    return json.dumps(type_dict, sort_keys=True)

def arg_types_from_json(arg_types_json: str) -> dict:
    """Reify the encoded argument types from the format produced by arg_types_to_json."""
    arg_types = json.loads(arg_types_json)
    return {name: type_from_dict(type_dict) for name, type_dict in arg_types.items()}
TypeEncoder = Callable[[type], str]

def maybe_encode_type(encode: str, typ: Union[typing.Type, None, str]) -> Union[None, str, bytes]:
    if typ is None:
        return None
    return encode(typ)
TypeDecoder = Callable[[str], type]

def maybe_decode_type(decode: Union[str, None], encoded: Union[str, None, bytes]) -> Union[None, str, bytes]:
    if encoded is None or encoded == 'null':
        return None
    return decode(encoded)
CallTraceRowT = TypeVar('CallTraceRowT', bound='CallTraceRow')

class CallTraceRow(CallTraceThunk):
    """A semi-structured call trace where each field has been json encoded."""

    def __init__(self, module: Union[typing.Mapping, typing.Type], qualname: Union[str, None, bool, typing.Type], arg_types: Any, return_type: typing.Type, yield_type: Any) -> None:
        self.module = module
        self.qualname = qualname
        self.arg_types = arg_types
        self.return_type = return_type
        self.yield_type = yield_type

    @classmethod
    def from_trace(cls: Union[typing.Callable, typing.Type], trace: Union[typing.Callable, typeline.tracing.CallTrace, dict]):
        module = trace.func.__module__
        qualname = trace.func.__qualname__
        arg_types = arg_types_to_json(trace.arg_types)
        return_type = maybe_encode_type(type_to_json, trace.return_type)
        yield_type = maybe_encode_type(type_to_json, trace.yield_type)
        return cls(module, qualname, arg_types, return_type, yield_type)

    def to_trace(self) -> CallTrace:
        function = get_func_in_module(self.module, self.qualname)
        arg_types = arg_types_from_json(self.arg_types)
        return_type = maybe_decode_type(type_from_json, self.return_type)
        yield_type = maybe_decode_type(type_from_json, self.yield_type)
        return CallTrace(function, arg_types, return_type, yield_type)

    def __eq__(self, other: Union[T, typing.Type, base.FieldFactory]) -> bool:
        if isinstance(other, CallTraceRow):
            return (self.module, self.qualname, self.arg_types, self.return_type, self.yield_type) == (other.module, other.qualname, other.arg_types, other.return_type, other.yield_type)
        return NotImplemented

def serialize_traces(traces: Any) -> typing.Generator:
    """Serialize an iterable of CallTraces to an iterable of CallTraceRow.

    Catches and logs exceptions, so a failure to serialize one CallTrace doesn't
    lose all traces.

    """
    for trace in traces:
        try:
            yield CallTraceRow.from_trace(trace)
        except Exception:
            logger.exception('Failed to serialize trace')