import time
from functools import wraps
from typing import Optional, Any, Callable, List, Dict, Union

from alerta.app import db


class Gauge:
    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, value: int = 0) -> None:
        self.group: str = group
        self.name: str = name
        self.title: Optional[str] = title
        self.description: Optional[str] = description
        self.type: str = 'gauge'
        self.value: int = value

    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        if format == 'prometheus':
            return (
                '# HELP alerta_{group}_{name} {description}\n'
                '# TYPE alerta_{group}_{name} gauge\n'
                'alerta_{group}_{name} {value}\n'.format(
                    group=self.group, name=self.name, description=self.description, value=self.value
                )
            )
        else:
            return {
                'group': self.group,
                'name': self.name,
                'title': self.title,
                'description': self.description,
                'type': self.type,
                'value': self.value
            }

    def __repr__(self) -> str:
        return 'Gauge(group={!r}, name={!r}, title={!r}, value={!r})'.format(
            self.group, self.name, self.title, self.value
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Gauge":
        return Gauge(
            group=doc.get('group'),
            name=doc.get('name'),
            title=doc.get('title', None),
            description=doc.get('description', None),
            value=doc.get('value', 0)
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Gauge":
        return Gauge(
            group=rec.group,
            name=rec.name,
            title=rec.title,
            description=rec.description,
            value=rec.value
        )

    @classmethod
    def from_db(cls, r: Any) -> Optional["Gauge"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return None

    def set(self, value: int) -> "Gauge":
        self.value = value
        result = db.set_gauge(self)
        return Gauge.from_db(result)  # type: ignore

    @classmethod
    def find_all(cls) -> List["Gauge"]:
        gauges = db.get_metrics(type='gauge')
        return [Gauge.from_db(gauge) for gauge in gauges if Gauge.from_db(gauge) is not None]  # type: ignore


class Counter:
    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: int = 0) -> None:
        self.group: str = group
        self.name: str = name
        self.title: Optional[str] = title
        self.description: Optional[str] = description
        self.type: str = 'counter'
        self.count: int = count

    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        if format == 'prometheus':
            return (
                '# HELP alerta_{group}_{name} {description}\n'
                '# TYPE alerta_{group}_{name} counter\n'
                'alerta_{group}_{name}_total {count}\n'.format(
                    group=self.group, name=self.name, description=self.description, count=self.count
                )
            )
        else:
            return {
                'group': self.group,
                'name': self.name,
                'title': self.title,
                'description': self.description,
                'type': self.type,
                'count': self.count
            }

    def __repr__(self) -> str:
        return 'Counter(group={!r}, name={!r}, title={!r}, count={!r})'.format(
            self.group, self.name, self.title, self.count
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Counter":
        return Counter(
            group=doc.get('group'),
            name=doc.get('name'),
            title=doc.get('title', None),
            description=doc.get('description', None),
            count=doc.get('count', 0)
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Counter":
        return Counter(
            group=rec.group,
            name=rec.name,
            title=rec.title,
            description=rec.description,
            count=rec.count
        )

    @classmethod
    def from_db(cls, r: Any) -> Optional["Counter"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return None

    def inc(self, count: int = 1) -> None:
        result = db.inc_counter(Counter(
            group=self.group,
            name=self.name,
            title=self.title,
            description=self.description,
            count=count
        ))
        c = Counter.from_db(result)  # type: ignore
        if c is not None:
            self.count = c.count

    @classmethod
    def find_all(cls) -> List["Counter"]:
        counters = db.get_metrics(type='counter')
        return [Counter.from_db(counter) for counter in counters if Counter.from_db(counter) is not None]  # type: ignore


class Timer:
    def __init__(self, group: str, name: str, title: Optional[str] = None, description: Optional[str] = None, count: int = 0, total_time: int = 0) -> None:
        self.group: str = group
        self.name: str = name
        self.title: Optional[str] = title
        self.description: Optional[str] = description
        self.type: str = 'timer'
        self.start: Optional[int] = None
        self.count: int = count
        self.total_time: int = total_time

    def serialize(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        if format == 'prometheus':
            return (
                '# HELP alerta_{group}_{name} {description}\n'
                '# TYPE alerta_{group}_{name} summary\n'
                'alerta_{group}_{name}_count {count}\n'
                'alerta_{group}_{name}_sum {total_time}\n'.format(
                    group=self.group, name=self.name, description=self.description, count=self.count, total_time=self.total_time
                )
            )
        else:
            return {
                'group': self.group,
                'name': self.name,
                'title': self.title,
                'description': self.description,
                'type': self.type,
                'count': self.count,
                'totalTime': self.total_time
            }

    def __repr__(self) -> str:
        return 'Timer(group={!r}, name={!r}, title={!r}, count={!r}, total_time={!r})'.format(
            self.group, self.name, self.title, self.count, self.total_time
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Timer":
        return Timer(
            group=doc.get('group'),
            name=doc.get('name'),
            title=doc.get('title', None),
            description=doc.get('description', None),
            count=doc.get('count', 0),
            total_time=doc.get('totalTime', 0)
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Timer":
        return Timer(
            group=rec.group,
            name=rec.name,
            title=rec.title,
            description=rec.description,
            count=rec.count,
            total_time=rec.total_time
        )

    @classmethod
    def from_db(cls, r: Any) -> Optional["Timer"]:
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
        result = db.update_timer(Timer(
            group=self.group,
            name=self.name,
            title=self.title,
            description=self.description,
            count=count,
            total_time=(self._time_in_millis() - start)
        ))
        t = Timer.from_db(result)  # type: ignore
        if t is not None:
            self.count = t.count
            self.total_time = t.total_time

    @classmethod
    def find_all(cls) -> List["Timer"]:
        timers = db.get_metrics(type='timer')
        return [Timer.from_db(timer) for timer in timers if Timer.from_db(timer) is not None]  # type: ignore


def timer(metric: Timer) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorated(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            ts: int = metric.start_timer()
            response: Any = f(*args, **kwargs)
            metric.stop_timer(ts)
            return response
        return wrapped
    return decorated