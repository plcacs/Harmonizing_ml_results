import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from mypy_extensions import TypedDict
from monkeytype.compat import is_any, is_generic, is_union, qualname_of_generic
from monkeytype.db.base import CallTraceThunk
from monkeytype.exceptions import InvalidTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType, NotImplementedType, is_typed_dict, mappingproxy
from monkeytype.util import get_func_in_module, get_name_in_module

logger = logging.getLogger(__name__)
TypeDict = Dict[str, Any]

def typed_dict_to_dict(typ: Type[TypedDict]) -> Dict[str, Any]:
    elem_types: Dict[str, Any] = {}
    for k, v in typ.__annotations__.items():
        elem_types[k] = type_to_dict(v)
    return {'module': typ.__module__, 'qualname': typ.__qualname__, 'elem_types': elem_types, 'is_typed_dict': True}

def type_to_dict(typ: Type) -> Dict[str, Any]:
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
    d: Dict[str, Any] = {'module': typ.__module__, 'qualname': qualname}
    elem_types = getattr(typ, '__args__', None)
    is_bare_generic = typ in {Dict, List, Tuple}
    if not is_bare_generic and elem_types is not None and is_generic(typ):
        if elem_types == ((),):
            elem_types = ()
        d['elem_types'] = [type_to_dict(t) for t in elem_types]
    return d

_HIDDEN_BUILTIN_TYPES: Dict[str, Type] = {'NoneType': NoneType, 'NotImplementedType': NotImplementedType, 'mappingproxy': mappingproxy}

def typed_dict_from_dict(d: Dict[str, Any]) -> Type[TypedDict]:
    return TypedDict(d['qualname'], {k: type_from_dict(v) for k, v in d['elem_types'].items()})

def type_from_dict(d: Dict[str, Any]) -> Type:
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

def type_to_json(typ: Type) -> str:
    type_dict = type_to_dict(typ)
    return json.dumps(type_dict, sort_keys=True)

def type_from_json(typ_json: str) -> Type:
    type_dict = json.loads(typ_json)
    return type_from_dict(type_dict)

def arg_types_to_json(arg_types: Dict[str, Type]) -> str:
    type_dict = {name: type_to_dict(typ) for name, typ in arg_types.items()}
    return json.dumps(type_dict, sort_keys=True)

def arg_types_from_json(arg_types_json: str) -> Dict[str, Type]:
    arg_types = json.loads(arg_types_json)
    return {name: type_from_dict(type_dict) for name, type_dict in arg_types.items()}

TypeEncoder = Callable[[Type], str]

def maybe_encode_type(encode: TypeEncoder, typ: Optional[Type]) -> Optional[str]:
    if typ is None:
        return None
    return encode(typ)

TypeDecoder = Callable[[str], Type]

def maybe_decode_type(decode: TypeDecoder, encoded: Optional[str]) -> Optional[Type]:
    if encoded is None or encoded == 'null':
        return None
    return decode(encoded)

CallTraceRowT = TypeVar('CallTraceRowT', bound='CallTraceRow')

class CallTraceRow(CallTraceThunk):
    def __init__(self, module: str, qualname: str, arg_types: str, return_type: Optional[str], yield_type: Optional[str]) -> None:
        self.module = module
        self.qualname = qualname
        self.arg_types = arg_types
        self.return_type = return_type
        self.yield_type = yield_type

    @classmethod
    def from_trace(cls: Type[CallTraceRowT], trace: CallTrace) -> CallTraceRowT:
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

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CallTraceRow):
            return (self.module, self.qualname, self.arg_types, self.return_type, self.yield_type) == (other.module, other.qualname, other.arg_types, other.return_type, other.yield_type)
        return NotImplemented

def serialize_traces(traces: Iterable[CallTrace]) -> Iterable[CallTraceRow]:
    for trace in traces:
        try:
            yield CallTraceRow.from_trace(trace)
        except Exception:
            logger.exception('Failed to serialize trace')
