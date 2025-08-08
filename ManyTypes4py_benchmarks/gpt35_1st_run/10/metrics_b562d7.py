    def __init__(self, group: str, name: str, title: str = None, description: str = None, value: int = 0) -> None:
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Union[str, int]]]:
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Gauge':
    @classmethod
    def from_record(cls, rec: Any) -> 'Gauge':
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> Optional['Gauge']:
    def set(self, value: int) -> 'Gauge':
    @classmethod
    def find_all(cls) -> List['Gauge']:

    def __init__(self, group: str, name: str, title: str = None, description: str = None, count: int = 0) -> None:
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Union[str, int]]]:
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Counter':
    @classmethod
    def from_record(cls, rec: Any) -> 'Counter':
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> Optional['Counter']:
    def inc(self, count: int = 1) -> None:
    @classmethod
    def find_all(cls) -> List['Counter']:

    def __init__(self, group: str, name: str, title: str = None, description: str = None, count: int = 0, total_time: int = 0) -> None:
    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Union[str, int]]]:
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Timer':
    @classmethod
    def from_record(cls, rec: Any) -> 'Timer':
    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple]) -> Optional['Timer']:
    def _time_in_millis(self) -> int:
    def start_timer(self) -> int:
    def stop_timer(self, start: int, count: int = 1) -> None:
    @classmethod
    def find_all(cls) -> List['Timer']:

    def decorated(f: Callable) -> Callable:
    def wrapped(*args, **kwargs) -> Any:
