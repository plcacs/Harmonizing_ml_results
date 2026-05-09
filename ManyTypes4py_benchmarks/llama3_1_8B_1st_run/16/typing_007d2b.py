import sys
from collections.abc import Callable
from os import PathLike
from typing import TYPE_CHECKING, AbstractSet, Any, Callable as TypingCallable, ClassVar, Dict, ForwardRef, Generator, Iterable, List, Mapping, NewType, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, _eval_type, cast, get_type_hints
from typing_extensions import Annotated, Final, Literal, NotRequired as TypedDictNotRequired, Required as TypedDictRequired

if TYPE_CHECKING:
    from pydantic.v1.fields import ModelField

def display_as_type(v: typing_base) -> str:
    if not isinstance(v, typing_base) and (not isinstance(v, WithArgsTypes)) and (not isinstance(v, type)):
        v = v.__class__
    if is_union(get_origin(v)):
        return f'Union[{", ".join(map(display_as_type, get_args(v)))}]'
    if isinstance(v, WithArgsTypes):
        return str(v).replace('typing.', '')
    try:
        return v.__name__
    except AttributeError:
        return str(v).replace('typing.', '')

def resolve_annotations(raw_annotations: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """
    Partially taken from typing.get_type_hints.

    Resolve string or ForwardRef annotations into type objects if possible.
    """
    base_globals = None
    if module_name:
        try:
            module = sys.modules[module_name]
        except KeyError:
            pass
        else:
            base_globals = module.__dict__
    annotations = {}
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

def update_field_forward_refs(field: ModelField, globalns: Dict[str, Any], localns: Dict[str, Any]) -> None:
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

def update_model_forward_refs(model: Any, fields: List[ModelField], json_encoders: Dict[Any, Any], localns: Dict[str, Any], exc_to_suppress: Tuple[Any, ...]) -> None:
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

def get_class(type_: Any) -> Any:
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
