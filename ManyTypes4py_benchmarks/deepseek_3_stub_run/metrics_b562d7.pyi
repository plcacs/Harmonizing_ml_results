import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import wraps
from alerta.app import db

F = TypeVar('F', bound=Callable[..., Any])

class Gauge:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    value: Union[int, float, None]
    
    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        value: Union[int, float] = 0
    ) -> None: ...
    
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Gauge': ...
    
    @classmethod
    def from_record(cls, rec: Any) -> 'Gauge': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional['Gauge']: ...
    
    def set(self, value: Union[int, float]) -> Optional['Gauge']: ...
    
    @classmethod
    def find_all(cls) -> List[Optional['Gauge']]: ...

class Counter:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    count: Union[int, float, None]
    
    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        count: Union[int, float] = 0
    ) -> None: ...
    
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Counter': ...
    
    @classmethod
    def from_record(cls, rec: Any) -> 'Counter': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional['Counter']: ...
    
    def inc(self, count: Union[int, float] = 1) -> None: ...
    
    @classmethod
    def find_all(cls) -> List[Optional['Counter']]: ...

class Timer:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    start: Optional[int]
    count: Union[int, float, None]
    total_time: Union[int, float, None]
    
    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        count: Union[int, float] = 0,
        total_time: Union[int, float] = 0
    ) -> None: ...
    
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Timer': ...
    
    @classmethod
    def from_record(cls, rec: Any) -> 'Timer': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional['Timer']: ...
    
    def _time_in_millis(self) -> int: ...
    
    def start_timer(self) -> int: ...
    
    def stop_timer(self, start: int, count: Union[int, float] = 1) -> None: ...
    
    @classmethod
    def find_all(cls) -> List[Optional['Timer']]: ...

def timer(metric: Timer) -> Callable[[F], F]: ...