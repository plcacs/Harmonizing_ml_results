from typing import Any, Optional, Union, List, Dict, Tuple

class Gauge:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    value: int

    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, value: int = 0) -> None: ...
    
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Gauge': ...
    
    @classmethod
    def from_record(cls, rec: Tuple[Any, ...]) -> 'Gauge': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...], Any]) -> Optional['Gauge']: ...
    
    def set(self, value: int) -> 'Gauge': ...
    
    @classmethod
    def find_all(cls) -> List['Gauge']: ...

class Counter:
    group: str
    name: str
    title: Optional[str]
    description: Optional[str]
    type: str
    count: int

    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: int = 0) -> None: ...
    
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Counter': ...
    
    @classmethod
    def from_record(cls, rec: Tuple[Any, ...]) -> 'Counter': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...], Any]) -> Optional['Counter']: ...
    
    def inc(self, count: int = 1) -> None: ...
    
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

    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: int = 0, total_time: int = 0) -> None: ...
    
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Timer': ...
    
    @classmethod
    def from_record(cls, rec: Tuple[Any, ...]) -> 'Timer': ...
    
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...], Any]) -> Optional['Timer']: ...
    
    def _time_in_millis(self) -> int: ...
    
    def start_timer(self) -> int: ...
    
    def stop_timer(self, start: int, count: int = 1) -> None: ...
    
    @classmethod
    def find_all(cls) -> List['Timer']: ...

def timer(metric: 'Timer') -> Any:
    def decorated(f: Any) -> Any:
        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> Any: ...
        return wrapped
    return decorated