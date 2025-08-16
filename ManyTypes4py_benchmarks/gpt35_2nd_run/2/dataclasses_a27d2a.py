from typing import Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union

def dataclass(*, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Any = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: Any = ...) -> Callable:
    ...

def set_validation(cls: Type, value: bool) -> Generator[Type, None, None]:
    ...

DataclassT = TypeVar('DataclassT', bound='Dataclass')
DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

class Dataclass:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def __get_validators__(cls):
        pass

    @classmethod
    def __validate__(cls, v):
        pass
