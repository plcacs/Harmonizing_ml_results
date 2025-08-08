from typing import TypeVar, Union, Tuple, Any, Collection, Callable, Optional, dict, Hashable, Set, Iterable, Sequence

KT = TypeVar('KT')
VT = TypeVar('VT', covariant=True)
VT1 = TypeVar('VT1', covariant=True)
VT2 = TypeVar('VT2', covariant=True)
R = TypeVar('R', covariant=True)
HashableT = TypeVar('HashableT', bound=Hashable)
T = TypeVar('T')

NestedDict = dict[KT, Union[VT, 'NestedDict[KT, VT]']]

def dict_to_flatdict(dct: dict) -> dict[tuple[KT, ...], VT]:
    ...

def flatdict_to_dict(dct: dict) -> NestedDict[KT, VT]:
    ...

def isiterable(obj: Any) -> bool:
    ...

def ensure_iterable(obj: T) -> Collection[T]:
    ...

def listrepr(objs: Iterable, sep: str = ' ') -> str:
    ...

def extract_instances(objects: Iterable, types: Union[type, Tuple[type, ...]] = object) -> Union[list, dict[type, list]]:
    ...

def batched_iterable(iterable: Iterable, size: int) -> Iterable[tuple]:
    ...

def remove_nested_keys(keys_to_remove: Iterable[HashableT], obj: Any) -> Any:
    ...

def distinct(iterable: Iterable, key: Optional[Callable[[Any], Hashable]] = None) -> Iterable:
    ...

def get_from_dict(dct: dict, keys: Union[str, Iterable], default: Any = None) -> Any:
    ...

def set_in_dict(dct: dict, keys: Union[str, Iterable], value: Any) -> None:
    ...

def deep_merge(dct: dict, merge: dict) -> dict:
    ...

def deep_merge_dicts(*dicts: dict) -> dict:
    ...
