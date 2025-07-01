import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Callable, Type
from alerta.app import db

T = TypeVar('T')

class Gauge:

    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, value: int = 0) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = 'gauge'
        self.value = value

    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        if format == 'prometheus':
            return '# HELP alerta_{group}_{name} {description}\n# TYPE alerta_{group}_{name} gauge\nalerta_{group}_{name} {value}\n'.format(group=self.group, name=self.name, description=self.description, value=self.value)
        else:
            return {'group': self.group, 'name': self.name, 'title': self.title, 'description': self.description, 'type': self.type, 'value': self.value}

    def __repr__(self) -> str:
        return 'Gauge(group={!r}, name={!r}, title={!r}, value={!r})'.format(self.group, self.name, self.title, self.value)

    @classmethod
    def from_document(cls: Type[T], doc: Dict[str, Any]) -> T:
        return Gauge(group=doc.get('group'), name=doc.get('name'), title=doc.get('title', None), description=doc.get('description', None), value=doc.get('value', None))

    @classmethod
    def from_record(cls: Type[T], rec: Any) -> T:
        return Gauge(group=rec.group, name=rec.name, title=rec.title, description=rec.description, value=rec.value)

    @classmethod
    def from_db(cls: Type[T], r: Union[Dict[str, Any], Tuple[Any, ...], None]) -> Optional[T]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return None

    def set(self, value: int) -> 'Gauge':
        self.value = value
        return Gauge.from_db(db.set_gauge(self))

    @classmethod
    def find_all(cls: Type[T]) -> List[T]:
        return [Gauge.from_db(gauge) for gauge in db.get_metrics(type='gauge')]

class Counter:

    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: int = 0) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = 'counter'
        self.count = count

    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        if format == 'prometheus':
            return '# HELP alerta_{group}_{name} {description}\n# TYPE alerta_{group}_{name} counter\nalerta_{group}_{name}_total {count}\n'.format(group=self.group, name=self.name, description=self.description, count=self.count)
        else:
            return {'group': self.group, 'name': self.name, 'title': self.title, 'description': self.description, 'type': self.type, 'count': self.count}

    def __repr__(self) -> str:
        return 'Counter(group={!r}, name={!r}, title={!r}, count={!r})'.format(self.group, self.name, self.title, self.count)

    @classmethod
    def from_document(cls: Type[T], doc: Dict[str, Any]) -> T:
        return Counter(group=doc.get('group'), name=doc.get('name'), title=doc.get('title', None), description=doc.get('description', None), count=doc.get('count', None))

    @classmethod
    def from_record(cls: Type[T], rec: Any) -> T:
        return Counter(group=rec.group, name=rec.name, title=rec.title, description=rec.description, count=rec.count)

    @classmethod
    def from_db(cls: Type[T], r: Union[Dict[str, Any], Tuple[Any, ...], None]) -> Optional[T]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return None

    def inc(self, count: int = 1) -> None:
        c = Counter.from_db(db.inc_counter(Counter(group=self.group, name=self.name, title=self.title, description=self.description, count=count)))
        self.count = c.count

    @classmethod
    def find_all(cls: Type[T]) -> List[T]:
        return [Counter.from_db(counter) for counter in db.get_metrics(type='counter')]

class Timer:

    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: int = 0, total_time: int = 0) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = 'timer'
        self.start = None
        self.count = count
        self.total_time = total_time

    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        if format == 'prometheus':
            return '# HELP alerta_{group}_{name} {description}\n# TYPE alerta_{group}_{name} summary\nalerta_{group}_{name}_count {count}\nalerta_{group}_{name}_sum {total_time}\n'.format(group=self.group, name=self.name, description=self.description, count=self.count, total_time=self.total_time)
        else:
            return {'group': self.group, 'name': self.name, 'title': self.title, 'description': self.description, 'type': self.type, 'count': self.count, 'totalTime': self.total_time}

    def __repr__(self) -> str:
        return 'Timer(group={!r}, name={!r}, title={!r}, count={!r}, total_time={!r})'.format(self.group, self.name, self.title, self.count, self.total_time)

    @classmethod
    def from_document(cls: Type[T], doc: Dict[str, Any]) -> T:
        return Timer(group=doc.get('group'), name=doc.get('name'), title=doc.get('title', None), description=doc.get('description', None), count=doc.get('count', None), total_time=doc.get('totalTime', None))

    @classmethod
    def from_record(cls: Type[T], rec: Any) -> T:
        return Timer(group=rec.group, name=rec.name, title=rec.title, description=rec.description, count=rec.count, total_time=rec.total_time)

    @classmethod
    def from_db(cls: Type[T], r: Union[Dict[str, Any], Tuple[Any, ...], None]) -> Optional[T]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return None

    def _time_in_millis(self) -> int:
        return int(round(time.time() * 1000))

    def start_timer(self) -> int:
        return self._time_in_millis()

    def stop_timer(self, start: int, count: int = 1) -> None:
        t = Timer.from_db(db.update_timer(Timer(group=self.group, name=self.name, title=self.title, description=self.description, count=count, total_time=self._time_in_millis() - start)))
        self.count = t.count
        self.total_time = t.total_time

    @classmethod
    def find_all(cls: Type[T]) -> List[T]:
        return [Timer.from_db(timer) for timer in db.get_metrics(type='timer')]

def timer(metric: Timer) -> Callable[[Callable[..., T]], Callable[..., T]]:

    def decorated(f: Callable[..., T]) -> Callable[..., T]:

        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            ts = metric.start_timer()
            response = f(*args, **kwargs)
            metric.stop_timer(ts)
            return response
        return wrapped
    return decorated
