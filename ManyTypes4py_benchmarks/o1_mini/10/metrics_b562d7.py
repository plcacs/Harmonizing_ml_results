import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from alerta.app import db


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
        title: Optional[str] = None,
        description: Optional[str] = None,
        value: float = 0.0,
    ) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = "gauge"
        self.value = value

    def serialize(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        if format == "prometheus":
            return (
                f"# HELP alerta_{self.group}_{self.name} {self.description}\n"
                f"# TYPE alerta_{self.group}_{self.name} gauge\n"
                f"alerta_{self.group}_{self.name} {self.value}\n"
            )
        else:
            return {
                "group": self.group,
                "name": self.name,
                "title": self.title,
                "description": self.description,
                "type": self.type,
                "value": self.value,
            }

    def __repr__(self) -> str:
        return (
            f"Gauge(group={self.group!r}, name={self.name!r}, title={self.title!r}, value={self.value!r})"
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Gauge":
        return cls(
            group=doc.get("group"),
            name=doc.get("name"),
            title=doc.get("title"),
            description=doc.get("description"),
            value=doc.get("value", 0.0),
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Gauge":
        return cls(
            group=rec.group,
            name=rec.name,
            title=rec.title,
            description=rec.description,
            value=rec.value,
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Any]) -> Optional["Gauge"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif hasattr(r, "group") and hasattr(r, "name"):
            return cls.from_record(r)
        else:
            return None

    def set(self, value: float) -> Optional["Gauge"]:
        self.value = value
        result = db.set_gauge(self)
        return Gauge.from_db(result)

    @classmethod
    def find_all(cls) -> List["Gauge"]:
        gauges = db.get_metrics(type="gauge")
        return [cls.from_db(gauge) for gauge in gauges if cls.from_db(gauge) is not None]


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
        title: Optional[str] = None,
        description: Optional[str] = None,
        count: int = 0,
    ) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = "counter"
        self.count = count

    def serialize(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        if format == "prometheus":
            return (
                f"# HELP alerta_{self.group}_{self.name} {self.description}\n"
                f"# TYPE alerta_{self.group}_{self.name} counter\n"
                f"alerta_{self.group}_{self.name}_total {self.count}\n"
            )
        else:
            return {
                "group": self.group,
                "name": self.name,
                "title": self.title,
                "description": self.description,
                "type": self.type,
                "count": self.count,
            }

    def __repr__(self) -> str:
        return (
            f"Counter(group={self.group!r}, name={self.name!r}, title={self.title!r}, count={self.count!r})"
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Counter":
        return cls(
            group=doc.get("group"),
            name=doc.get("name"),
            title=doc.get("title"),
            description=doc.get("description"),
            count=doc.get("count", 0),
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Counter":
        return cls(
            group=rec.group,
            name=rec.name,
            title=rec.title,
            description=rec.description,
            count=rec.count,
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Any]) -> Optional["Counter"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif hasattr(r, "group") and hasattr(r, "name"):
            return cls.from_record(r)
        else:
            return None

    def inc(self, count: int = 1) -> None:
        updated_counter = Counter(
            group=self.group,
            name=self.name,
            title=self.title,
            description=self.description,
            count=count,
        )
        result = db.inc_counter(updated_counter)
        updated = Counter.from_db(result)
        if updated:
            self.count = updated.count

    @classmethod
    def find_all(cls) -> List["Counter"]:
        counters = db.get_metrics(type="counter")
        return [cls.from_db(counter) for counter in counters if cls.from_db(counter) is not None]


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
        title: Optional[str] = None,
        description: Optional[str] = None,
        count: int = 0,
        total_time: int = 0,
    ) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = "timer"
        self.start = None
        self.count = count
        self.total_time = total_time

    def serialize(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        if format == "prometheus":
            return (
                f"# HELP alerta_{self.group}_{self.name} {self.description}\n"
                f"# TYPE alerta_{self.group}_{self.name} summary\n"
                f"alerta_{self.group}_{self.name}_count {self.count}\n"
                f"alerta_{self.group}_{self.name}_sum {self.total_time}\n"
            )
        else:
            return {
                "group": self.group,
                "name": self.name,
                "title": self.title,
                "description": self.description,
                "type": self.type,
                "count": self.count,
                "totalTime": self.total_time,
            }

    def __repr__(self) -> str:
        return (
            f"Timer(group={self.group!r}, name={self.name!r}, title={self.title!r}, "
            f"count={self.count!r}, total_time={self.total_time!r})"
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Timer":
        return cls(
            group=doc.get("group"),
            name=doc.get("name"),
            title=doc.get("title"),
            description=doc.get("description"),
            count=doc.get("count", 0),
            total_time=doc.get("totalTime", 0),
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Timer":
        return cls(
            group=rec.group,
            name=rec.name,
            title=rec.title,
            description=rec.description,
            count=rec.count,
            total_time=rec.total_time,
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Any]) -> Optional["Timer"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif hasattr(r, "group") and hasattr(r, "name"):
            return cls.from_record(r)
        else:
            return None

    def _time_in_millis(self) -> int:
        return int(round(time.time() * 1000))

    def start_timer(self) -> int:
        return self._time_in_millis()

    def stop_timer(self, start: int, count: int = 1) -> None:
        elapsed = self._time_in_millis() - start
        updated_timer = Timer(
            group=self.group,
            name=self.name,
            title=self.title,
            description=self.description,
            count=count,
            total_time=elapsed,
        )
        result = db.update_timer(updated_timer)
        updated = Timer.from_db(result)
        if updated:
            self.count = updated.count
            self.total_time = updated.total_time

    @classmethod
    def find_all(cls) -> List["Timer"]:
        timers = db.get_metrics(type="timer")
        return [cls.from_db(timer) for timer in timers if cls.from_db(timer) is not None]


def timer(metric: Timer) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorated(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            ts = metric.start_timer()
            response = f(*args, **kwargs)
            metric.stop_timer(ts)
            return response

        return wrapped

    return decorated
