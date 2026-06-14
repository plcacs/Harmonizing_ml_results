from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

_F = TypeVar('_F', bound=Callable[..., Any])

class Gauge:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    value: Any

    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        value: Any = ...,
    ) -> None: ...
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    def __repr__(self) -> str: ...
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> Gauge: ...
    @classmethod
    def from_record(cls, rec: Any) -> Gauge: ...
    @classmethod
    def from_db(cls, r: Union[dict, tuple, Any]) -> Optional[Gauge]: ...
    def set(self, value: Any) -> Optional[Gauge]: ...
    @classmethod
    def find_all(cls) -> List[Optional[Gauge]]: ...

class Counter:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    count: int

    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        count: int = ...,
    ) -> None: ...
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    def __repr__(self) -> str: ...
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> Counter: ...
    @classmethod
    def from_record(cls, rec: Any) -> Counter: ...
    @classmethod
    def from_db(cls, r: Union[dict, tuple, Any]) -> Optional[Counter]: ...
    def inc(self, count: int = ...) -> None: ...
    @classmethod
    def find_all(cls) -> List[Optional[Counter]]: ...

class Timer:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    start: Optional[int]
    count: int
    total_time: int

    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        count: int = ...,
        total_time: int = ...,
    ) -> None: ...
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    def __repr__(self) -> str: ...
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> Timer: ...
    @classmethod
    def from_record(cls, rec: Any) -> Timer: ...
    @classmethod
    def from_db(cls, r: Union[dict, tuple, Any]) -> Optional[Timer]: ...
    def _time_in_millis(self) -> int: ...
    def start_timer(self) -> int: ...
    def stop_timer(self, start: int, count: int = ...) -> None: ...
    @classmethod
    def find_all(cls) -> List[Optional[Timer]]: ...

def timer(metric: Timer) -> Callable[[_F], _F]: ...