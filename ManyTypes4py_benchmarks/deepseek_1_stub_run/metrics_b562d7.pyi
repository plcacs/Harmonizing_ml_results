```python
import time
from typing import Any, Optional, List, Union, Dict, Callable, TypeVar
from functools import wraps

F = TypeVar('F', bound=Callable[..., Any])

class Gauge:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    value: float
    
    def __init__(
        self,
        group: str,
        name: str,
        title: Optional[str] = ...,
        description: Optional[str] = ...,
        value: float = ...
    ) -> None: ...
    
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Gauge': ...
    
    @classmethod
    def from_record(cls, rec: Any) -> 'Gauge': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], tuple, Any]) -> Optional['Gauge']: ...
    
    def set(self, value: float) -> Optional['Gauge']: ...
    
    @classmethod
    def find_all(cls) -> List['Gauge']: ...

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
        count: int = ...
    ) -> None: ...
    
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Counter': ...
    
    @classmethod
    def from_record(cls, rec: Any) -> 'Counter': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], tuple, Any]) -> Optional['Counter']: ...
    
    def inc(self, count: int = ...) -> None: ...
    
    @classmethod
    def find_all(cls) -> List['Counter']: ...

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
        total_time: int = ...
    ) -> None: ...
    
    def serialize(self, format: str = ...) -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Timer': ...
    
    @classmethod
    def from_record(cls, rec: Any) -> 'Timer': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], tuple, Any]) -> Optional['Timer']: ...
    
    def _time_in_millis(self) -> int: ...
    
    def start_timer(self) -> int: ...
    
    def stop_timer(self, start: int, count: int = ...) -> None: ...
    
    @classmethod
    def find_all(cls) -> List['Timer']: ...

def timer(metric: Timer) -> Callable[[F], F]: ...
```