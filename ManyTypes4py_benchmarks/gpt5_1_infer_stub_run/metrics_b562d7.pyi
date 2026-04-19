from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class Gauge:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    value: Optional[Union[int, float]]

    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        value: Union[int, float, None] = ...,
    ) -> None: ...
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    def __repr__(self) -> str: ...
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Gauge": ...
    @classmethod
    def from_record(cls, rec: Any) -> "Gauge": ...
    @classmethod
    def from_db(cls, r: object) -> Optional["Gauge"]: ...
    def set(self, value: Union[int, float, None]) -> Optional["Gauge"]: ...
    @classmethod
    def find_all(cls) -> List[Optional["Gauge"]]: ...


class Counter:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    count: Optional[int]

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
    def from_document(cls, doc: Dict[str, Any]) -> "Counter": ...
    @classmethod
    def from_record(cls, rec: Any) -> "Counter": ...
    @classmethod
    def from_db(cls, r: object) -> Optional["Counter"]: ...
    def inc(self, count: int = ...) -> None: ...
    @classmethod
    def find_all(cls) -> List[Optional["Counter"]]: ...


class Timer:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    start: Optional[int]
    count: Optional[int]
    total_time: Optional[int]

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
    def from_document(cls, doc: Dict[str, Any]) -> "Timer": ...
    @classmethod
    def from_record(cls, rec: Any) -> "Timer": ...
    @classmethod
    def from_db(cls, r: object) -> Optional["Timer"]: ...
    def _time_in_millis(self) -> int: ...
    def start_timer(self) -> int: ...
    def stop_timer(self, start: int, count: int = ...) -> None: ...
    @classmethod
    def find_all(cls) -> List[Optional["Timer"]]: ...


def timer(metric: Timer) -> Callable[[Callable[P, T]], Callable[P, T]]: ...