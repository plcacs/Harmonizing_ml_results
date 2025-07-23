import io
import itertools
import types
import warnings
from collections import OrderedDict
from collections.abc import Callable, Collection, Generator, Hashable, Iterable, Iterator, Sequence, Set
from dataclasses import fields, is_dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast, overload
from unittest.mock import Mock
import pydantic
from typing_extensions import TypeAlias, TypeVar
from prefect.utilities.annotations import BaseAnnotation as BaseAnnotation
from prefect.utilities.annotations import Quote as Quote
from prefect.utilities.annotations import quote as quote

if TYPE_CHECKING:
    pass

class AutoEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name: str, *args: Any, **kwargs: Any) -> str:
        return name

    @staticmethod
    def auto() -> auto:
        return auto()

    def __repr__(self) -> str:
        return f'{type(self).__name__}.{self.value}'

KT = TypeVar('KT')
VT = TypeVar('VT', infer_variance=True)
VT1 = TypeVar('VT1', infer_variance=True)
VT2 = TypeVar('VT2', infer_variance=True)
R = TypeVar('R', infer_variance=True)
NestedDict = dict[KT, Union[VT, 'NestedDict[KT, VT]']]
HashableT = TypeVar('HashableT', bound=Hashable)

def dict_to_flatdict(dct: dict[KT, VT]) -> dict[tuple[KT, ...], VT]:
    def flatten(dct: dict[KT, VT], _parent: tuple[KT, ...] = ()) -> Iterator[tuple[tuple[KT, ...], VT]]:
        parent = _parent or ()
        for k, v in dct.items():
            k_parent = (*parent, k)
            if isinstance(v, dict) and v:
                yield from flatten(cast(NestedDict[KT, VT], v), _parent=k_parent)
            else:
                yield (k_parent, cast(VT, v))
    type_ = cast(type[dict[tuple[KT, ...], VT]], type(dct))
    return type_(flatten(dct))

def flatdict_to_dict(dct: dict[tuple[KT, ...], VT]) -> NestedDict[KT, VT]:
    type_ = cast(type[NestedDict[KT, VT]], type(dct))

    def new(type_: type[NestedDict[KT, VT]] = type_) -> NestedDict[KT, VT]:
        return type_()
    result = new()
    for key_tuple, value in dct.items():
        current = result
        *prefix_keys, last_key = key_tuple
        for prefix_key in prefix_keys:
            try:
                current = cast(NestedDict[KT, VT], current[prefix_key])
            except KeyError:
                new_dict = current[prefix_key] = new()
                current = new_dict
        current[last_key] = value
    return result

T = TypeVar('T')

def isiterable(obj: Any) -> bool:
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return not isinstance(obj, (str, bytes, io.IOBase))

def ensure_iterable(obj: Union[Sequence[T], Set[T], T]) -> Collection[T]:
    if isinstance(obj, Sequence) or isinstance(obj, Set):
        return cast(Collection[T], obj)
    obj = cast(T, obj)
    return [obj]

def listrepr(objs: Iterable[Any], sep: str = ' ') -> str:
    return sep.join((repr(obj) for obj in objs))

def extract_instances(objects: Iterable[Any], types: Union[type, tuple[type, ...]] = object) -> Union[list[Any], dict[type, list[Any]]]:
    types_collection = ensure_iterable(types)
    ret: dict[type, list[Any]] = {}
    for o in objects:
        for type_ in types_collection:
            if isinstance(o, type_):
                ret.setdefault(type_, []).append(o)
    if len(types_collection) == 1:
        [type_] = types_collection
        return ret[type_]
    return ret

def batched_iterable(iterable: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, size))
        if not batch:
            break
        yield batch

class StopVisiting(BaseException):
    pass

@overload
def visit_collection(expr: Any, visit_fn: Callable[[Any], Any], *, return_data: bool = ..., max_depth: int = ..., context: Optional[dict[str, VT]] = ..., remove_annotations: bool = ..., _seen: Optional[Set[int]] = ...) -> Optional[Any]:
    ...

@overload
def visit_collection(expr: Any, visit_fn: Callable[[Any, dict[str, VT]], Any], *, return_data: bool = ..., max_depth: int = ..., context: dict[str, VT], remove_annotations: bool = ..., _seen: Optional[Set[int]] = ...) -> Optional[Any]:
    ...

def visit_collection(expr: Any, visit_fn: Union[Callable[[Any], Any], Callable[[Any, dict[str, VT]], Any]], *, return_data: bool = False, max_depth: int = -1, context: Optional[dict[str, VT]] = None, remove_annotations: bool = False, _seen: Optional[Set[int]] = None) -> Optional[Any]:
    if _seen is None:
        _seen = set()
    if context is not None:
        _callback = cast(Callable[[Any, dict[str, VT]], Any], visit_fn)

        def visit_nested(expr: Any) -> Optional[Any]:
            return visit_collection(expr, _callback, return_data=return_data, remove_annotations=remove_annotations, max_depth=max_depth - 1, context=context.copy(), _seen=_seen)

        def visit_expression(expr: Any) -> Any:
            return _callback(expr, context)
    else:
        _callback = cast(Callable[[Any], Any], visit_fn)

        def visit_nested(expr: Any) -> Optional[Any]:
            return visit_collection(expr, _callback, return_data=return_data, remove_annotations=remove_annotations, max_depth=max_depth - 1, _seen=_seen)

        def visit_expression(expr: Any) -> Any:
            return _callback(expr)
    try:
        result = visit_expression(expr)
    except StopVisiting:
        max_depth = 0
        result = expr
    if return_data:
        expr = result
    if max_depth == 0 or id(expr) in _seen:
        return result if return_data else None
    else:
        _seen.add(id(expr))
    result = expr
    if isinstance(expr, (types.GeneratorType, types.AsyncGeneratorType)):
        pass
    elif isinstance(expr, Mock):
        pass
    elif isinstance(expr, BaseAnnotation):
        annotated = cast(BaseAnnotation[Any], expr)
        if context is not None:
            context['annotation'] = cast(VT, annotated)
        unwrapped = annotated.unwrap()
        value = visit_nested(unwrapped)
        if return_data:
            if remove_annotations:
                result = value
            elif value is not unwrapped:
                result = annotated.rewrap(value)
    elif isinstance(expr, (list, tuple, set)):
        seq = cast(Union[list[Any], tuple[Any], set[Any]], expr)
        items = [visit_nested(o) for o in seq]
        if return_data:
            modified = any((item is not orig for item, orig in zip(items, seq)))
            if modified:
                result = type(seq)(items)
    elif isinstance(expr, (dict, OrderedDict)):
        mapping = cast(dict[Any, Any], expr)
        items = [(visit_nested(k), visit_nested(v)) for k, v in mapping.items()]
        if return_data:
            modified = any((k1 is not k2 or v1 is not v2 for (k1, v1), (k2, v2) in zip(items, mapping.items())))
            if modified:
                result = type(mapping)(items)
    elif is_dataclass(expr) and (not isinstance(expr, type)):
        expr_fields = fields(expr)
        values = [visit_nested(getattr(expr, f.name)) for f in expr_fields]
        if return_data:
            modified = any((getattr(expr, f.name) is not v for f, v in zip(expr_fields, values)))
            if modified:
                result = replace(expr, **{f.name: v for f, v in zip(expr_fields, values)})
    elif isinstance(expr, pydantic.BaseModel):
        model_fields = expr.model_fields_set.union(expr.model_fields.keys())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            updated_data = {field: visit_nested(getattr(expr, field)) for field in model_fields}
        if return_data:
            modified = any((getattr(expr, field) is not updated_data[field] for field in model_fields))
            if modified:
                model_instance = expr.model_construct(_fields_set=expr.model_fields_set, **updated_data)
                for private_attr in expr.__private_attributes__:
                    setattr(model_instance, private_attr, getattr(expr, private_attr))
                result = model_instance
    if return_data:
        return result

@overload
def remove_nested_keys(keys_to_remove: Collection[HashableT], obj: NestedDict[HashableT, VT]) -> NestedDict[HashableT, VT]:
    ...

@overload
def remove_nested_keys(keys_to_remove: Collection[HashableT], obj: Any) -> Any:
    ...

def remove_nested_keys(keys_to_remove: Collection[HashableT], obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {key: remove_nested_keys(keys_to_remove, value) for key, value in cast(NestedDict[HashableT, VT], obj).items() if key not in keys_to_remove}

@overload
def distinct(iterable: Iterable[T], key: None = None) -> Iterator[T]:
    ...

@overload
def distinct(iterable: Iterable[T], key: Callable[[T], Hashable]) -> Iterator[T]:
    ...

def distinct(iterable: Iterable[T], key: Optional[Callable[[T], Hashable]] = None) -> Iterator[T]:
    def _key(__i: T) -> Hashable:
        return __i
    if key is not None:
        _key = cast(Callable[[T], Hashable], key)
    seen: set[Hashable] = set()
    for item in iterable:
        if _key(item) in seen:
            continue
        seen.add(_key(item))
        yield item

@overload
def get_from_dict(dct: Union[NestedDict[str, VT], list[VT]], keys: Union[str, list[Union[str, int]]], default: None = None) -> Optional[VT]:
    ...

@overload
def get_from_dict(dct: Union[NestedDict[str, VT], list[VT]], keys: Union[str, list[Union[str, int]]], default: VT) -> VT:
    ...

def get_from_dict(dct: Union[NestedDict[str, VT], list[VT]], keys: Union[str, list[Union[str, int]]], default: Optional[VT] = None) -> Optional[VT]:
    if isinstance(keys, str):
        keys = keys.replace('[', '.').replace(']', '').split('.')
    value: Any = dct
    try:
        for key in keys:
            try:
                key = int(key)
            except ValueError:
                pass
            value = value[key]
        return cast(VT, value)
    except (TypeError, KeyError, IndexError):
        return default

def set_in_dict(dct: NestedDict[str, VT], keys: Union[str, list[str]], value: VT) -> None:
    if isinstance(keys, str):
        keys = keys.replace('[', '.').replace(']', '').split('.')
    for k in keys[:-1]:
        if not isinstance(dct.get(k, {}), dict):
            raise TypeError(f'Key path exists and contains a non-dict value: {keys}')
        if k not in dct:
            dct[k] = {}
        dct = cast(NestedDict[str, VT], dct[k])
    dct[keys[-1]] = value

def deep_merge(dct: NestedDict[str, VT1], merge: NestedDict[str, VT2]) -> NestedDict[str, Union[VT1, VT2]]:
    result = dct.copy()
    for key, value in merge.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(cast(NestedDict[str, VT1], result[key]), cast(NestedDict[str, VT2], value))
        else:
            result[key] = cast(Union[VT2, NestedDict[str, VT2]], value)
    return result

def deep_merge_dicts(*dicts: NestedDict[str, VT]) -> NestedDict[str, VT]:
    result: NestedDict[str, VT] = {}
    for dictionary in dicts:
        result = deep_merge(result, dictionary)
    return result
