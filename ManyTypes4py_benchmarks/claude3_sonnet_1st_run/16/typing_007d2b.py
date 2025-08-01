import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import TYPE_CHECKING, AbstractSet, Any, Callable as TypingCallable, ClassVar, Dict, ForwardRef, Generator, Iterable, List, Mapping, NewType, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, _eval_type, cast, get_type_hints
from typing_extensions import Annotated, Final, Literal, NotRequired as TypedDictNotRequired, Required as TypedDictRequired

try:
    from typing import _TypingBase as typing_base
except ImportError:
    from typing import _Final as typing_base

try:
    from typing import GenericAlias as TypingGenericAlias
except ImportError:
    TypingGenericAlias = ()

try:
    from types import UnionType as TypesUnionType
except ImportError:
    TypesUnionType = ()

if sys.version_info < (3, 9):
    def evaluate_forwardref(type_: ForwardRef, globalns: Dict[str, Any], localns: Optional[Dict[str, Any]]) -> Any:
        return type_._evaluate(globalns, localns)
else:
    def evaluate_forwardref(type_: ForwardRef, globalns: Dict[str, Any], localns: Optional[Dict[str, Any]]) -> Any:
        return cast(Any, type_)._evaluate(globalns, localns, recursive_guard=set())

if sys.version_info < (3, 9):
    get_all_type_hints = get_type_hints
else:
    def get_all_type_hints(obj: Any, globalns: Optional[Dict[str, Any]] = None, localns: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return get_type_hints(obj, globalns, localns, include_extras=True)

_T = TypeVar('_T')
AnyCallable = TypingCallable[..., Any]
NoArgAnyCallable = TypingCallable[[], Any]
AnyArgTCallable = TypingCallable[..., _T]

AnnotatedTypeNames: Set[str] = {'AnnotatedMeta', '_AnnotatedAlias'}
LITERAL_TYPES: Set[Any] = {Literal}

if hasattr(typing, 'Literal'):
    LITERAL_TYPES.add(typing.Literal)

if sys.version_info < (3, 8):
    def get_origin(t: Any) -> Optional[Any]:
        if type(t).__name__ in AnnotatedTypeNames:
            return cast(Type[Any], Annotated)
        return getattr(t, '__origin__', None)
else:
    from typing import get_origin as _typing_get_origin

    def get_origin(tp: Any) -> Optional[Any]:
        """
        We can't directly use `typing.get_origin` since we need a fallback to support
        custom generic classes like `ConstrainedList`
        It should be useless once https://github.com/cython/cython/issues/3537 is
        solved and https://github.com/pydantic/pydantic/pull/1753 is merged.
        """
        if type(tp).__name__ in AnnotatedTypeNames:
            return cast(Type[Any], Annotated)
        return _typing_get_origin(tp) or getattr(tp, '__origin__', None)

if sys.version_info < (3, 8):
    from typing import _GenericAlias

    def get_args(t: Any) -> Tuple[Any, ...]:
        """Compatibility version of get_args for python 3.7.

        Mostly compatible with the python 3.8 `typing` module version
        and able to handle almost all use cases.
        """
        if type(t).__name__ in AnnotatedTypeNames:
            return t.__args__ + t.__metadata__
        if isinstance(t, _GenericAlias):
            res = t.__args__
            if t.__origin__ is Callable and res and (res[0] is not Ellipsis):
                res = (list(res[:-1]), res[-1])
            return res
        return getattr(t, '__args__', ())
else:
    from typing import get_args as _typing_get_args

    def _generic_get_args(tp: Any) -> Tuple[Any, ...]:
        """
        In python 3.9, `typing.Dict`, `typing.List`, ...
        do have an empty `__args__` by default (instead of the generic ~T for example).
        In order to still support `Dict` for example and consider it as `Dict[Any, Any]`,
        we retrieve the `_nparams` value that tells us how many parameters it needs.
        """
        if hasattr(tp, '_nparams'):
            return (Any,) * tp._nparams
        try:
            if tp == Tuple[()] or (sys.version_info >= (3, 9) and tp == tuple[()]):
                return ((),)
        except TypeError:
            pass
        return ()

    def get_args(tp: Any) -> Tuple[Any, ...]:
        """Get type arguments with all substitutions performed.

        For unions, basic simplifications used by Union constructor are performed.
        Examples::
            get_args(Dict[str, int]) == (str, int)
            get_args(int) == ()
            get_args(Union[int, Union[T, int], str][int]) == (int, str)
            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
            get_args(Callable[[], T][int]) == ([], int)
        """
        if type(tp).__name__ in AnnotatedTypeNames:
            return tp.__args__ + tp.__metadata__
        return _typing_get_args(tp) or getattr(tp, '__args__', ()) or _generic_get_args(tp)

if sys.version_info < (3, 9):
    def convert_generics(tp: Any) -> Any:
        """Python 3.9 and older only supports generics from `typing` module.
        They convert strings to ForwardRef automatically.

        Examples::
            typing.List['Hero'] == typing.List[ForwardRef('Hero')]
        """
        return tp
else:
    from typing import _UnionGenericAlias
    from typing_extensions import _AnnotatedAlias

    def convert_generics(tp: Any) -> Any:
        """
        Recursively searches for `str` type hints and replaces them with ForwardRef.

        Examples::
            convert_generics(list['Hero']) == list[ForwardRef('Hero')]
            convert_generics(dict['Hero', 'Team']) == dict[ForwardRef('Hero'), ForwardRef('Team')]
            convert_generics(typing.Dict['Hero', 'Team']) == typing.Dict[ForwardRef('Hero'), ForwardRef('Team')]
            convert_generics(list[str | 'Hero'] | int) == list[str | ForwardRef('Hero')] | int
        """
        origin = get_origin(tp)
        if not origin or not hasattr(tp, '__args__'):
            return tp
        args = get_args(tp)
        if origin is Annotated:
            return _AnnotatedAlias(convert_generics(args[0]), args[1:])
        converted = tuple((ForwardRef(arg) if isinstance(arg, str) and isinstance(tp, TypingGenericAlias) else convert_generics(arg) for arg in args))
        if converted == args:
            return tp
        elif isinstance(tp, TypingGenericAlias):
            return TypingGenericAlias(origin, converted)
        elif isinstance(tp, TypesUnionType):
            return _UnionGenericAlias(origin, converted)
        else:
            try:
                setattr(tp, '__args__', converted)
            except AttributeError:
                pass
            return tp

if sys.version_info < (3, 10):
    def is_union(tp: Any) -> bool:
        return tp is Union
    WithArgsTypes = (TypingGenericAlias,)
else:
    import types
    import typing

    def is_union(tp: Any) -> bool:
        return tp is Union or tp is types.UnionType
    WithArgsTypes = (typing._GenericAlias, types.GenericAlias, types.UnionType)

StrPath = Union[str, PathLike]

if TYPE_CHECKING:
    from pydantic.v1.fields import ModelField
    TupleGenerator = Generator[Tuple[str, Any], None, None]
    DictStrAny = Dict[str, Any]
    DictAny = Dict[Any, Any]
    SetStr = Set[str]
    ListStr = List[str]
    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictIntStrAny = Dict[IntStr, Any]
    MappingIntStrAny = Mapping[IntStr, Any]
    CallableGenerator = Generator[AnyCallable, None, None]
    ReprArgs = Sequence[Tuple[Optional[str], Any]]
    MYPY = False
    if MYPY:
        AnyClassMethod = classmethod[Any]
    else:
        AnyClassMethod = classmethod[Any, Any, Any]

__all__ = ('AnyCallable', 'NoArgAnyCallable', 'NoneType', 'is_none_type', 'display_as_type', 'resolve_annotations', 'is_callable_type', 'is_literal_type', 'all_literal_values', 'is_namedtuple', 'is_typeddict', 'is_typeddict_special', 'is_new_type', 'new_type_supertype', 'is_classvar', 'is_finalvar', 'update_field_forward_refs', 'update_model_forward_refs', 'TupleGenerator', 'DictStrAny', 'DictAny', 'SetStr', 'ListStr', 'IntStr', 'AbstractSetIntStr', 'DictIntStrAny', 'CallableGenerator', 'ReprArgs', 'AnyClassMethod', 'CallableGenerator', 'WithArgsTypes', 'get_args', 'get_origin', 'get_sub_types', 'typing_base', 'get_all_type_hints', 'is_union', 'StrPath', 'MappingIntStrAny')

NoneType = None.__class__
NONE_TYPES = (None, NoneType, Literal[None])

if sys.version_info < (3, 8):
    def is_none_type(type_: Any) -> bool:
        return type_ in NONE_TYPES
elif sys.version_info[:2] == (3, 8):
    def is_none_type(type_: Any) -> bool:
        for none_type in NONE_TYPES:
            if type_ is none_type:
                return True
        if is_literal_type(type_):
            return all_literal_values(type_) == (None,)
        return False
else:
    def is_none_type(type_: Any) -> bool:
        return type_ in NONE_TYPES

def display_as_type(v: Any) -> str:
    if not isinstance(v, typing_base) and (not isinstance(v, WithArgsTypes)) and (not isinstance(v, type)):
        v = v.__class__
    if is_union(get_origin(v)):
        return f'Union[{', '.join(map(display_as_type, get_args(v)))}]'
    if isinstance(v, WithArgsTypes):
        return str(v).replace('typing.', '')
    try:
        return v.__name__
    except AttributeError:
        return str(v).replace('typing.', '')

def resolve_annotations(raw_annotations: Dict[str, Any], module_name: Optional[str]) -> Dict[str, Any]:
    """
    Partially taken from typing.get_type_hints.

    Resolve string or ForwardRef annotations into type objects if possible.
    """
    base_globals: Optional[Dict[str, Any]] = None
    if module_name:
        try:
            module = sys.modules[module_name]
        except KeyError:
            pass
        else:
            base_globals = module.__dict__
    annotations: Dict[str, Any] = {}
    for name, value in raw_annotations.items():
        if isinstance(value, str):
            if (3, 10) > sys.version_info >= (3, 9, 8) or sys.version_info >= (3, 10, 1):
                value = ForwardRef(value, is_argument=False, is_class=True)
            else:
                value = ForwardRef(value, is_argument=False)
        try:
            if sys.version_info >= (3, 13):
                value = _eval_type(value, base_globals, None, type_params=())
            else:
                value = _eval_type(value, base_globals, None)
        except NameError:
            pass
        annotations[name] = value
    return annotations

def is_callable_type(type_: Any) -> bool:
    return type_ is Callable or get_origin(type_) is Callable

def is_literal_type(type_: Any) -> bool:
    return Literal is not None and get_origin(type_) in LITERAL_TYPES

def literal_values(type_: Any) -> Tuple[Any, ...]:
    return get_args(type_)

def all_literal_values(type_: Any) -> Tuple[Any, ...]:
    """
    This method is used to retrieve all Literal values as
    Literal can be used recursively (see https://www.python.org/dev/peps/pep-0586)
    e.g. `Literal[Literal[Literal[1, 2, 3], "foo"], 5, None]`
    """
    if not is_literal_type(type_):
        return (type_,)
    values = literal_values(type_)
    return tuple((x for value in values for x in all_literal_values(value)))

def is_namedtuple(type_: Any) -> bool:
    """
    Check if a given class is a named tuple.
    It can be either a `typing.NamedTuple` or `collections.namedtuple`
    """
    from pydantic.v1.utils import lenient_issubclass
    return lenient_issubclass(type_, tuple) and hasattr(type_, '_fields')

def is_typeddict(type_: Any) -> bool:
    """
    Check if a given class is a typed dict (from `typing` or `typing_extensions`)
    In 3.10, there will be a public method (https://docs.python.org/3.10/library/typing.html#typing.is_typeddict)
    """
    from pydantic.v1.utils import lenient_issubclass
    return lenient_issubclass(type_, dict) and hasattr(type_, '__total__')

def _check_typeddict_special(type_: Any) -> bool:
    return type_ is TypedDictRequired or type_ is TypedDictNotRequired

def is_typeddict_special(type_: Any) -> bool:
    """
    Check if type is a TypedDict special form (Required or NotRequired).
    """
    return _check_typeddict_special(type_) or _check_typeddict_special(get_origin(type_))

test_type = NewType('test_type', str)

def is_new_type(type_: Any) -> bool:
    """
    Check whether type_ was created using typing.NewType
    """
    return isinstance(type_, test_type.__class__) and hasattr(type_, '__supertype__')

def new_type_supertype(type_: Any) -> Any:
    while hasattr(type_, '__supertype__'):
        type_ = type_.__supertype__
    return type_

def _check_classvar(v: Any) -> bool:
    if v is None:
        return False
    return v.__class__ == ClassVar.__class__ and getattr(v, '_name', None) == 'ClassVar'

def _check_finalvar(v: Any) -> bool:
    """
    Check if a given type is a `typing.Final` type.
    """
    if v is None:
        return False
    return v.__class__ == Final.__class__ and (sys.version_info < (3, 8) or getattr(v, '_name', None) == 'Final')

def is_classvar(ann_type: Any) -> bool:
    if _check_classvar(ann_type) or _check_classvar(get_origin(ann_type)):
        return True
    if ann_type.__class__ == ForwardRef and ann_type.__forward_arg__.startswith('ClassVar['):
        return True
    return False

def is_finalvar(ann_type: Any) -> bool:
    return _check_finalvar(ann_type) or _check_finalvar(get_origin(ann_type))

def update_field_forward_refs(field: 'ModelField', globalns: Dict[str, Any], localns: Optional[Dict[str, Any]]) -> None:
    """
    Try to update ForwardRefs on fields based on this ModelField, globalns and localns.
    """
    prepare = False
    if field.type_.__class__ == ForwardRef:
        prepare = True
        field.type_ = evaluate_forwardref(field.type_, globalns, localns or None)
    if field.outer_type_.__class__ == ForwardRef:
        prepare = True
        field.outer_type_ = evaluate_forwardref(field.outer_type_, globalns, localns or None)
    if prepare:
        field.prepare()
    if field.sub_fields:
        for sub_f in field.sub_fields:
            update_field_forward_refs(sub_f, globalns=globalns, localns=localns)
    if field.discriminator_key is not None:
        field.prepare_discriminated_union_sub_fields()

def update_model_forward_refs(
    model: Type[Any], 
    fields: List['ModelField'], 
    json_encoders: Dict[Any, Any], 
    localns: Optional[Dict[str, Any]], 
    exc_to_suppress: Tuple[Type[Exception], ...] = ()
) -> None:
    """
    Try to update model fields ForwardRefs based on model and localns.
    """
    if model.__module__ in sys.modules:
        globalns = sys.modules[model.__module__].__dict__.copy()
    else:
        globalns = {}
    globalns.setdefault(model.__name__, model)
    for f in fields:
        try:
            update_field_forward_refs(f, globalns=globalns, localns=localns)
        except exc_to_suppress:
            pass
    for key in set(json_encoders.keys()):
        if isinstance(key, str):
            fr = ForwardRef(key)
        elif isinstance(key, ForwardRef):
            fr = key
        else:
            continue
        try:
            new_key = evaluate_forwardref(fr, globalns, localns or None)
        except exc_to_suppress:
            continue
        json_encoders[new_key] = json_encoders.pop(key)

def get_class(type_: Any) -> Optional[Union[bool, Type[Any]]]:
    """
    Tries to get the class of a Type[T] annotation. Returns True if Type is used
    without brackets. Otherwise returns None.
    """
    if type_ is type:
        return True
    if get_origin(type_) is None:
        return None
    args = get_args(type_)
    if not args or not isinstance(args[0], type):
        return True
    else:
        return args[0]

def get_sub_types(tp: Any) -> List[Any]:
    """
    Return all the types that are allowed by type `tp`
    `tp` can be a `Union` of allowed types or an `Annotated` type
    """
    origin = get_origin(tp)
    if origin is Annotated:
        return get_sub_types(get_args(tp)[0])
    elif is_union(origin):
        return [x for t in get_args(tp) for x in get_sub_types(t)]
    else:
        return [tp]
