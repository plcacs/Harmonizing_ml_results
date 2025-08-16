from typing import Optional, Union, List, Dict

class Gauge:
    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, value: Union[int, float] = 0) -> None:
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Union[str, int, float]]]:
    @classmethod
    def from_document(cls, doc: Dict[str, Optional[Union[str, int, float]]]) -> 'Gauge':
    @classmethod
    def from_record(cls, rec: 'Record') -> 'Gauge':
    @classmethod
    def from_db(cls, r: Union[Dict[str, Optional[Union[str, int, float]]], Tuple]) -> Optional['Gauge']:
    def set(self, value: Union[int, float]) -> Optional['Gauge']:
    @classmethod
    def find_all(cls) -> List['Gauge']:

class Counter:
    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: Union[int, float] = 0) -> None:
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Union[str, int, float]]]:
    @classmethod
    def from_document(cls, doc: Dict[str, Optional[Union[str, int, float]]]) -> 'Counter':
    @classmethod
    def from_record(cls, rec: 'Record') -> 'Counter':
    @classmethod
    def from_db(cls, r: Union[Dict[str, Optional[Union[str, int, float]]], Tuple]) -> Optional['Counter']:
    def inc(self, count: Union[int, float] = 1) -> None:
    @classmethod
    def find_all(cls) -> List['Counter']:

class Timer:
    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: Union[int, float] = 0, total_time: Union[int, float] = 0) -> None:
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Union[str, int, float]]]:
    @classmethod
    def from_document(cls, doc: Dict[str, Optional[Union[str, int, float]]]) -> 'Timer':
    @classmethod
    def from_record(cls, rec: 'Record') -> 'Timer':
    @classmethod
    def from_db(cls, r: Union[Dict[str, Optional[Union[str, int, float]]], Tuple]) -> Optional['Timer']:
    def _time_in_millis(self) -> int:
    def start_timer(self) -> int:
    def stop_timer(self, start: int, count: Union[int, float] = 1) -> None:
    @classmethod
    def find_all(cls) -> List['Timer']:

def timer(metric: Timer) -> Callable:
