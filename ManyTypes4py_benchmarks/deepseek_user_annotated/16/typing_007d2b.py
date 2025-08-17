import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import (  # type: ignore
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable as TypingCallable,
    ClassVar,
    Dict,
    ForwardRef,
    Generator,
    Iterable,
    List,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    _eval_type,
    cast,
    get_type_hints,
)

from typing_extensions import (
    Annotated,
    Final,
    Literal,
    NotRequired as TypedDictNotRequired,
    Required as TypedDictRequired,
)

try:
    from typing import _TypingBase as typing_base  # type: ignore
except ImportError:
    from typing import _Final as typing_base  # type: ignore

try:
    from typing import GenericAlias as TypingGenericAlias  # type: ignore
except ImportError:
    # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
    TypingGenericAlias = ()

try:
    from types import UnionType as TypesUnionType  # type: ignore
except ImportError:
    # python < 3.10 does not have UnionType (str | int, byte | bool and so on)
    TypesUnionType = ()


if sys.version_info < (3, 9):
    def evaluate_forwardref(type_: ForwardRef, globalns: Any, localns: Any) -> Any:
        return type_._evaluate(globalns, localns)
else:
    def evaluate_forwardref(type_: ForwardRef, globalns: Any, localns: Any) -> Any:
        return cast(Any, type_)._evaluate(globalns, localns, recursive_guard=set())


if sys.version_info < (3, 9):
    get_all_type_hints: Callable[..., Any] = get_type_hints
else:
    def get_all_type_hints(obj: Any, globalns: Any = None, localns: Any = None) -> Any:
        return get_type_hints(obj, globalns, localns, include_extras=True)


_T = TypeVar('_T')

AnyCallable: Type[Callable[..., Any]] = TypingCallable[..., Any]
NoArgAnyCallable: Type[Callable[[], Any]] = TypingCallable[[], Any]
AnyArgTCallable: Type[Callable[..., _T]] = TypingCallable[..., _T]

AnnotatedTypeNames: Set[str] = {'AnnotatedMeta', '_AnnotatedAlias'}

LITERAL_TYPES: Set[Any] = {Literal}
if hasattr(typing, 'Literal'):
    LITERAL_TYPES.add(typing.Literal)


if sys.version_info < (3, 8):
    def get_origin(t: Type[Any]) -> Optional[Type[Any]]:
        if type(t).__name__ in AnnotatedTypeNames:
            return cast(Type[Any], Annotated)
        return getattr(t, '__origin__', None)
else:
    from typing import get_origin as _typing_get_origin

    def get_origin(tp: Type[Any]) -> Optional[Type[Any]]:
        if type(tp).__name__ in AnnotatedTypeNames:
            return cast(Type[Any], Annotated)
        return _typing_get_origin(tp) or getattr(tp, '__origin__', None)


if sys.version_info < (3, 8):
    from typing import _GenericAlias

    def get_args(t: Type[Any]) -> Tuple[Any, ...]:
        if type(t).__name__ in AnnotatedTypeNames:
            return t.__args__ + t.__metadata__
        if isinstance(t, _GenericAlias):
            res = t.__args__
            if t.__origin__ is Callable and res and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res
        return getattr(t, '__args__', ())
else:
    from typing import get_args as _typing_get_args

    def _generic_get_args(tp: Type[Any]) -> Tuple[Any, ...]:
        if hasattr(tp, '_nparams'):
            return (Any,) * tp._nparams
        try:
            if tp == Tuple[()] or sys.version_info >= (3, 9) and tp == tuple[()]:  # type: ignore[misc]
                return ((),)
        except TypeError:
            pass
        return ()

    def get_args(tp: Type[Any]) -> Tuple[Any, ...]:
        if type(tp).__name__ in AnnotatedTypeNames:
            return tp.__args__ + tp.__metadata__
        return _typing_get_args(tp) or getattr(tp, '__args__', ()) or _generic_get_args(tp)


if sys.version_info < (3, 9):
    def convert_generics(tp: Type[Any]) -> Type[Any]:
        return tp
else:
    from typing import _UnionGenericAlias  # type: ignore
    from typing_extensions import _AnnotatedAlias

    def convert_generics(tp: Type[Any]) -> Type[Any]:
        origin = get_origin(tp)
        if not origin or not hasattr(tp, '__args__'):
            return tp

        args = get_args(tp)

        if origin is Annotated:
            return _AnnotatedAlias(convert_generics(args[0]), args[1:])

        converted = tuple(
            ForwardRef(arg) if isinstance(arg, str) and isinstance(tp, TypingGenericAlias) else convert_generics(arg)
            for arg in args
        )

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
    def is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is Union

    WithArgsTypes: Tuple[Type[Any], ...] = (TypingGenericAlias,)
else:
    import types
    import typing

    def is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is Union or tp is types.UnionType  # noqa: E721

    WithArgsTypes: Tuple[Type[Any], ...] = (typing._GenericAlias, types.GenericAlias, types.UnionType)


StrPath: Type[Union[str, PathLike]] = Union[str, PathLike]


if TYPE_CHECKING:
    from pydantic.v1.fields import ModelField

    TupleGenerator: Type[Generator[Tuple[str, Any], None, None]] = Generator[Tuple[str, Any], None, None]
    DictStrAny: Type[Dict[str, Any]] = Dict[str, Any]
    DictAny: Type[Dict[Any, Any]] = Dict[Any, Any]
    SetStr: Type[Set[str]] = Set[str]
    ListStr: Type[List[str]] = List[str]
    IntStr: Type[Union[int, str]] = Union[int, str]
    AbstractSetIntStr: Type[AbstractSet[IntStr]] = AbstractSet[IntStr]
    DictIntStrAny: Type[Dict[IntStr, Any]] = Dict[IntStr, Any]
    MappingIntStrAny: Type[Mapping[IntStr, Any]] = Mapping[IntStr, Any]
    CallableGenerator: Type[Generator[AnyCallable, None, None]] = Generator[AnyCallable, None, None]
    ReprArgs: Type[Sequence[Tuple[Optional[str], Any]]] = Sequence[Tuple[Optional[str], Any]]

    MYPY: bool = False
    if MYPY:
        AnyClassMethod: Type[classmethod[Any]] = classmethod[Any]
    else:
        AnyClassMethod: Type[classmethod[Any, Any, Any]] = classmethod[Any, Any, Any]


NoneType: Type[None] = None.__class__

NONE_TYPES: Tuple[Any, Any, Any] = (None, NoneType, Literal[None])


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


def display_as_type(v: Type[Any]) -> str:
    if not isinstance(v, typing_base) and not isinstance(v, WithArgsTypes) and not isinstance(v, type):
        v = v.__class__

    if is_union(get_origin(v)):
        return f'Union[{", ".join(map(display_as_type, get_args(v)))}]'

    if isinstance(v, WithArgsTypes):
        return str(v).replace('typing.', '')

    try:
        return v.__name__
    except AttributeError:
        return str(v).replace('typing.', '')


def resolve_annotations(raw_annotations: Dict[str, Type[Any]], module_name: Optional[str]) -> Dict[str, Type[Any]]:
    base_globals: Optional[Dict[str, Any]] = None
    if module_name:
        try:
            module = sys.modules[module_name]
        except KeyError:
            pass
        else:
            base_globals = module.__dict__.copy()

    annotations: Dict[str, Type[Any]] = {}
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


def is_callable_type(type_: Type[Any]) -> bool:
    return type_ is Callable or get_origin(type_) is Callable


def is_literal_type(type_: Type[Any]) -> bool:
    return Literal is not None and get_origin(type_) in LITERAL_TYPES


def literal_values(type_: Type[Any]) -> Tuple[Any, ...]:
    return get_args(type_)


def all_literal_values(type_: Type[Any]) -> Tuple[Any, ...]:
    if not is_literal_type(type_):
        return (type_,)

    values = literal_values(type_)
    return tuple(x for value in values for x in all_literal_values(value))


def is_namedtuple(type_: Type[Any]) -> bool:
    from pydantic.v1.utils import lenient_issubclass

    return lenient_issubclass(type_, tuple) and hasattr(type_, '_fields')


def is_typeddict(type_: Type[Any]) -> bool:
    from pydantic.v1.utils import lenient_issubclass

    return lenient_issubclass(type_, dict) and hasattr(type_, '__total__')


def _check_typeddict_special(type_: Any) -> bool:
    return type_ is TypedDictRequired or type_ is TypedDictNotRequired


def is_typeddict_special(type_: Any) -> bool:
    return _check_typeddict_special(type_) or _check_typeddict_special(get_origin(type_))


def is_new_type(type_: Type[Any]) -> bool:
    return isinstance(type_, test_type.__class__) and hasattr(type_, '__supertype__')  # type: ignore


def new_type_supertype(type_: Type[Any]) -> Type[Any]:
    while hasattr(type_, '__supertype__'):
        type_ = type_.__supertype__
    return type_


def _check_classvar(v: Optional[Type[Any]]) -> bool:
    if v is None:
        return False
    return v.__class__ == ClassVar.__class__ and getattr(v, '_name', None) == 'ClassVar'


def _check_finalvar(v: Optional[Type[Any]]) -> bool:
    if v is None:
        return False
    return v.__class__ == Final.__class__ and (sys.version_info < (3, 8) or getattr(v, '_name', None) == 'Final')


def is_classvar(ann_type: Type[Any]) -> bool:
    if _check_classvar(ann_type) or _check_classvar(get_origin(ann_type)):
        return True
    if ann_type.__class__ == ForwardRef and ann_type.__forward_arg__.startswith('ClassVar['):
        return True
    return False


def is_finalvar(ann_type: Type[Any]) -> bool:
    return _check_finalvar(ann_type) or _check_finalvar(get_origin(ann_type))


def update_field_forward_refs(field: 'ModelField', globalns: Any, localns: Any) -> None:
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
    fields: Iterable['ModelField'],
    json_encoders: Dict[Union[Type[Any], str, ForwardRef], AnyCallable],
    localns: 'DictStrAny',
    exc_to_suppress: Tuple[Type[BaseException], ...] = (),
) -> None:
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
            fr: ForwardRef = ForwardRef(key)
        elif isinstance(key, ForwardRef):
            fr = key
        else:
            continue

        try:
            new_key = evaluate_forwardref(fr, globalns, localns or None)
        except exc_to_suppress:
            continue

        json_encoders[new_key] = json_encoders.pop(key)


def get_class(type_: Type[Any]) -> Union[None, bool, Type[Any]]:
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
    origin = get_origin(tp)
    if origin is Annotated:
        return get_sub_types(get_args(tp)[0])
    elif is_union(origin):
        return [x for t in get_args(tp) for x in get_sub_types(t)]
    else:
        return [tp]


__all__ = (
    'AnyCallable',
    'NoArgAnyCallable',
    'NoneType',
    'is_none_type',
    'display_as_type',
    'resolve_annotations',
    'is_callable_type',
    'is_literal_type',
    'all_literal_values',
    'is_namedtuple',
    'is_typeddict',
    'is_typeddict_special',
    'is_new_type',
    'new_type_supertype',
    'is_classvar',
    'is_finalvar',
    'update_field_forward_refs',
    'update_model_forward_refs',
    'TupleGenerator',
    'DictStrAny',
    'DictAny',
    'SetStr',
    'ListStr',
    'IntStr',
    'AbstractSetIntStr',
    'DictIntStrAny',
    'CallableGenerator',
    'ReprArgs',
    'AnyClassMethod',
    'CallableGenerator',
    'WithArgsTypes',
    'get_args',
    'get_origin',
    'get_sub_types',
    'typing_base',
    'get_all_type_hints',
    'is_union',
    'StrPath',
    'MappingIntStrAny',
)
