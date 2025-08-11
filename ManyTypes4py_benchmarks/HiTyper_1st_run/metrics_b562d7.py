import time
from functools import wraps
from alerta.app import db

class Gauge:

    def __init__(self, group: Union[str, None], name: Union[str, None], title: Union[None, str, int]=None, description: Union[None, str]=None, value=0) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = 'gauge'
        self.value = value

    def serialize(self, format: typing.Text='json') -> Union[str, dict[typing.Text, ]]:
        if format == 'prometheus':
            return '# HELP alerta_{group}_{name} {description}\n# TYPE alerta_{group}_{name} gauge\nalerta_{group}_{name} {value}\n'.format(group=self.group, name=self.name, description=self.description, value=self.value)
        else:
            return {'group': self.group, 'name': self.name, 'title': self.title, 'description': self.description, 'type': self.type, 'value': self.value}

    def __repr__(self) -> str:
        return 'Gauge(group={!r}, name={!r}, title={!r}, value={!r})'.format(self.group, self.name, self.title, self.value)

    @classmethod
    def from_document(cls: Union[dict, str, dict[str, str]], doc: Union[dict[str, typing.Any], dict]) -> Timer:
        return Gauge(group=doc.get('group'), name=doc.get('name'), title=doc.get('title', None), description=doc.get('description', None), value=doc.get('value', None))

    @classmethod
    def from_record(cls: Union[dict, list[dict], str, None], rec: typing.Mapping) -> Timer:
        return Gauge(group=rec.group, name=rec.name, title=rec.title, description=rec.description, value=rec.value)

    @classmethod
    def from_db(cls: Any, r: Union[dict, tuple, str, None]) -> None:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return

    def set(self, value: Union[str, float, T]):
        self.value = value
        return Gauge.from_db(db.set_gauge(self))

    @classmethod
    def find_all(cls: Union[str, typing.Callable[typing.Any, T], bool]) -> list:
        return [Gauge.from_db(gauge) for gauge in db.get_metrics(type='gauge')]

class Counter:

    def __init__(self, group: Union[str, None], name: Union[str, None], title: Union[None, str, int]=None, description: Union[None, str]=None, count: int=0) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = 'counter'
        self.count = count

    def serialize(self, format: typing.Text='json') -> Union[str, dict[typing.Text, ]]:
        if format == 'prometheus':
            return '# HELP alerta_{group}_{name} {description}\n# TYPE alerta_{group}_{name} counter\nalerta_{group}_{name}_total {count}\n'.format(group=self.group, name=self.name, description=self.description, count=self.count)
        else:
            return {'group': self.group, 'name': self.name, 'title': self.title, 'description': self.description, 'type': self.type, 'count': self.count}

    def __repr__(self) -> str:
        return 'Counter(group={!r}, name={!r}, title={!r}, count={!r})'.format(self.group, self.name, self.title, self.count)

    @classmethod
    def from_document(cls: Union[dict, str, dict[str, str]], doc: Union[dict[str, typing.Any], dict]) -> Timer:
        return Counter(group=doc.get('group'), name=doc.get('name'), title=doc.get('title', None), description=doc.get('description', None), count=doc.get('count', None))

    @classmethod
    def from_record(cls: Union[dict, list[dict], str, None], rec: typing.Mapping) -> Timer:
        return Counter(group=rec.group, name=rec.name, title=rec.title, description=rec.description, count=rec.count)

    @classmethod
    def from_db(cls: Any, r: Union[dict, tuple, str, None]) -> None:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return

    def inc(self, count: int=1) -> None:
        c = Counter.from_db(db.inc_counter(Counter(group=self.group, name=self.name, title=self.title, description=self.description, count=count)))
        self.count = c.count

    @classmethod
    def find_all(cls: Union[str, typing.Callable[typing.Any, T], bool]) -> list:
        return [Counter.from_db(counter) for counter in db.get_metrics(type='counter')]

class Timer:

    def __init__(self, group: Union[str, None], name: Union[str, None], title: Union[None, str, int]=None, description: Union[None, str]=None, count: int=0, total_time: int=0) -> None:
        self.group = group
        self.name = name
        self.title = title
        self.description = description
        self.type = 'timer'
        self.start = None
        self.count = count
        self.total_time = total_time

    def serialize(self, format: typing.Text='json') -> Union[str, dict[typing.Text, ]]:
        if format == 'prometheus':
            return '# HELP alerta_{group}_{name} {description}\n# TYPE alerta_{group}_{name} summary\nalerta_{group}_{name}_count {count}\nalerta_{group}_{name}_sum {total_time}\n'.format(group=self.group, name=self.name, description=self.description, count=self.count, total_time=self.total_time)
        else:
            return {'group': self.group, 'name': self.name, 'title': self.title, 'description': self.description, 'type': self.type, 'count': self.count, 'totalTime': self.total_time}

    def __repr__(self) -> str:
        return 'Timer(group={!r}, name={!r}, title={!r}, count={!r}, total_time={!r})'.format(self.group, self.name, self.title, self.count, self.total_time)

    @classmethod
    def from_document(cls: Union[dict, str, dict[str, str]], doc: Union[dict[str, typing.Any], dict]) -> Timer:
        return Timer(group=doc.get('group'), name=doc.get('name'), title=doc.get('title', None), description=doc.get('description', None), count=doc.get('count', None), total_time=doc.get('totalTime', None))

    @classmethod
    def from_record(cls: Union[dict, list[dict], str, None], rec: typing.Mapping) -> Timer:
        return Timer(group=rec.group, name=rec.name, title=rec.title, description=rec.description, count=rec.count, total_time=rec.total_time)

    @classmethod
    def from_db(cls: Any, r: Union[dict, tuple, str, None]) -> None:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            return

    def _time_in_millis(self) -> int:
        return int(round(time.time() * 1000))

    def start_timer(self) -> Union[bool, str]:
        return self._time_in_millis()

    def stop_timer(self, start: int, count: int=1) -> None:
        t = Timer.from_db(db.update_timer(Timer(group=self.group, name=self.name, title=self.title, description=self.description, count=count, total_time=self._time_in_millis() - start)))
        self.count = t.count
        self.total_time = t.total_time

    @classmethod
    def find_all(cls: Union[str, typing.Callable[typing.Any, T], bool]) -> list:
        return [Timer.from_db(timer) for timer in db.get_metrics(type='timer')]

def timer(metric: Union[float, typing.Iterable[float]]):

    def decorated(f: Any):

        @wraps(f)
        def wrapped(*args, **kwargs):
            ts = metric.start_timer()
            response = f(*args, **kwargs)
            metric.stop_timer(ts)
            return response
        return wrapped
    return decorated