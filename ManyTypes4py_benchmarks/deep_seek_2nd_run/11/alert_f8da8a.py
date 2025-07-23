import os
import platform
import sys
from datetime import datetime
from typing import Optional, Any, Dict, List, Tuple, Union, TypeVar, Type, cast
from uuid import UUID, uuid4
from flask import current_app, g
from alerta.app import alarm_model, db
from alerta.database.base import Query
from alerta.models.enums import ChangeType
from alerta.models.history import History, RichHistory
from alerta.models.note import Note
from alerta.utils.format import DateTime
from alerta.utils.hooks import status_change_hook
from alerta.utils.response import absolute_url

JSON = Dict[str, Any]
NoneType = type(None)
T = TypeVar('T', bound='Alert')

class Alert:

    def __init__(
        self,
        resource: str,
        event: str,
        *,
        id: Optional[str] = None,
        environment: Optional[str] = None,
        severity: Optional[str] = None,
        correlate: Optional[List[str]] = None,
        status: Optional[str] = None,
        service: Optional[List[str]] = None,
        group: Optional[str] = None,
        value: Optional[str] = None,
        text: Optional[str] = None,
        tags: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        origin: Optional[str] = None,
        event_type: Optional[str] = None,
        create_time: Optional[datetime] = None,
        timeout: Optional[int] = None,
        raw_data: Optional[str] = None,
        customer: Optional[str] = None,
        duplicate_count: Optional[int] = None,
        repeat: Optional[bool] = None,
        previous_severity: Optional[str] = None,
        trend_indication: Optional[str] = None,
        receive_time: Optional[datetime] = None,
        last_receive_id: Optional[str] = None,
        last_receive_time: Optional[datetime] = None,
        update_time: Optional[datetime] = None,
        history: Optional[List[History]] = None,
        **kwargs: Any
    ) -> None:
        if not resource:
            raise ValueError('Missing mandatory value for "resource"')
        if not event:
            raise ValueError('Missing mandatory value for "event"')
        if any(['.' in key for key in kwargs.get('attributes', dict()).keys()]) or any(['$' in key for key in kwargs.get('attributes', dict()).keys()]):
            raise ValueError('Attribute keys must not contain "." or "$"')
        if isinstance(kwargs.get('value', None), int):
            kwargs['value'] = str(kwargs['value'])
        for attr in ['create_time', 'receive_time', 'last_receive_time']:
            if not isinstance(kwargs.get(attr), (datetime, NoneType)):
                raise ValueError(f"Attribute '{attr}' must be datetime type")
        timeout = kwargs.get('timeout') if kwargs.get('timeout') is not None else current_app.config['ALERT_TIMEOUT']
        try:
            timeout = int(timeout)
        except ValueError:
            raise ValueError(f"Could not convert 'timeout' value of '{timeout}' to an integer")
        if timeout < 0:
            raise ValueError(f"Invalid negative 'timeout' value ({timeout})")
        self.id = id or str(uuid4())
        self.resource = resource
        self.event = event
        self.environment = environment or ''
        self.severity = severity or alarm_model.DEFAULT_NORMAL_SEVERITY
        self.correlate = correlate or list()
        if self.correlate and event not in self.correlate:
            self.correlate.append(event)
        self.status = status or alarm_model.DEFAULT_STATUS
        self.service = service or list()
        self.group = group or 'Misc'
        self.value = value
        self.text = text or ''
        self.tags = tags or list()
        self.attributes = attributes or dict()
        self.origin = origin or f'{os.path.basename(sys.argv[0])}/{platform.uname()[1]}'
        self.event_type = event_type or kwargs.get('type', None) or 'exceptionAlert'
        self.create_time = create_time or datetime.utcnow()
        self.timeout = timeout
        self.raw_data = raw_data
        self.customer = customer
        self.duplicate_count = duplicate_count
        self.repeat = repeat
        self.previous_severity = previous_severity
        self.trend_indication = trend_indication
        self.receive_time = receive_time or datetime.utcnow()
        self.last_receive_id = last_receive_id
        self.last_receive_time = last_receive_time
        self.update_time = update_time
        self.history = history or list()

    @classmethod
    def parse(cls: Type[T], json: JSON) -> T:
        if not isinstance(json.get('correlate', []), list):
            raise ValueError('correlate must be a list')
        if not isinstance(json.get('service', []), list):
            raise ValueError('service must be a list')
        if not isinstance(json.get('tags', []), list):
            raise ValueError('tags must be a list')
        if not isinstance(json.get('attributes', {}), dict):
            raise ValueError('attributes must be a JSON object')
        if not isinstance(json.get('timeout') if json.get('timeout', None) is not None else 0, int):
            raise ValueError('timeout must be an integer')
        if json.get('customer', None) == '':
            raise ValueError('customer must not be an empty string')
        return cls(
            id=json.get('id'),
            resource=json.get('resource'),
            event=json.get('event'),
            environment=json.get('environment'),
            severity=json.get('severity'),
            correlate=json.get('correlate', list()),
            status=json.get('status'),
            service=json.get('service', list()),
            group=json.get('group'),
            value=json.get('value'),
            text=json.get('text'),
            tags=json.get('tags', list()),
            attributes=json.get('attributes', dict()),
            origin=json.get('origin'),
            event_type=json.get('type'),
            create_time=DateTime.parse(json['createTime']) if 'createTime' in json else None,
            timeout=json.get('timeout'),
            raw_data=json.get('rawData'),
            customer=json.get('customer')
        )

    @property
    def serialize(self) -> JSON:
        return {
            'id': self.id,
            'href': absolute_url('/alert/' + self.id),
            'resource': self.resource,
            'event': self.event,
            'environment': self.environment,
            'severity': self.severity,
            'correlate': self.correlate,
            'status': self.status,
            'service': self.service,
            'group': self.group,
            'value': self.value,
            'text': self.text,
            'tags': self.tags,
            'attributes': self.attributes,
            'origin': self.origin,
            'type': self.event_type,
            'createTime': self.create_time,
            'timeout': self.timeout,
            'rawData': self.raw_data,
            'customer': self.customer,
            'duplicateCount': self.duplicate_count,
            'repeat': self.repeat,
            'previousSeverity': self.previous_severity,
            'trendIndication': self.trend_indication,
            'receiveTime': self.receive_time,
            'lastReceiveId': self.last_receive_id,
            'lastReceiveTime': self.last_receive_time,
            'updateTime': self.update_time,
            'history': [h.serialize for h in sorted(self.history, key=lambda x: x.update_time)]
        }

    def get_id(self, short: bool = False) -> str:
        return self.id[:8] if short else self.id

    def get_body(self, history: bool = True) -> JSON:
        body = self.serialize
        body.update({key: DateTime.iso8601(body[key]) for key in ['createTime', 'lastReceiveTime', 'receiveTime', 'updateTime'] if body[key]})
        if not history:
            body['history'] = []
        return body

    def __repr__(self) -> str:
        return f'Alert(id={self.id!r}, environment={self.environment!r}, resource={self.resource!r}, event={self.event!r}, severity={self.severity!r}, status={self.status!r}, customer={self.customer!r})'

    @classmethod
    def from_document(cls: Type[T], doc: Dict[str, Any]) -> T:
        return cls(
            id=doc.get('id', None) or doc.get('_id'),
            resource=doc.get('resource'),
            event=doc.get('event'),
            environment=doc.get('environment'),
            severity=doc.get('severity'),
            correlate=doc.get('correlate', list()),
            status=doc.get('status'),
            service=doc.get('service', list()),
            group=doc.get('group'),
            value=doc.get('value'),
            text=doc.get('text'),
            tags=doc.get('tags', list()),
            attributes=doc.get('attributes', dict()),
            origin=doc.get('origin'),
            event_type=doc.get('type'),
            create_time=doc.get('createTime'),
            timeout=doc.get('timeout'),
            raw_data=doc.get('rawData'),
            customer=doc.get('customer'),
            duplicate_count=doc.get('duplicateCount'),
            repeat=doc.get('repeat'),
            previous_severity=doc.get('previousSeverity'),
            trend_indication=doc.get('trendIndication'),
            receive_time=doc.get('receiveTime'),
            last_receive_id=doc.get('lastReceiveId'),
            last_receive_time=doc.get('lastReceiveTime'),
            update_time=doc.get('updateTime'),
            history=[History.from_db(h) for h in doc.get('history', list())]
        )

    @classmethod
    def from_record(cls: Type[T], rec: Any) -> T:
        return cls(
            id=rec.id,
            resource=rec.resource,
            event=rec.event,
            environment=rec.environment,
            severity=rec.severity,
            correlate=rec.correlate,
            status=rec.status,
            service=rec.service,
            group=rec.group,
            value=rec.value,
            text=rec.text,
            tags=rec.tags,
            attributes=dict(rec.attributes),
            origin=rec.origin,
            event_type=rec.type,
            create_time=rec.create_time,
            timeout=rec.timeout,
            raw_data=rec.raw_data,
            customer=rec.customer,
            duplicate_count=rec.duplicate_count,
            repeat=rec.repeat,
            previous_severity=rec.previous_severity,
            trend_indication=rec.trend_indication,
            receive_time=rec.receive_time,
            last_receive_id=rec.last_receive_id,
            last_receive_time=rec.last_receive_time,
            update_time=getattr(rec, 'update_time'),
            history=[History.from_db(h) for h in rec.history]
        )

    @classmethod
    def from_db(cls: Type[T], r: Union[Dict[str, Any], Tuple[Any, ...]]) -> T:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        raise ValueError(f"Unsupported type for from_db: {type(r)}")

    def is_duplicate(self) -> Optional['Alert']:
        """Return duplicate alert or None"""
        return Alert.from_db(db.is_duplicate(self))

    def is_correlated(self) -> Optional['Alert']:
        """Return correlated alert or None"""
        return Alert.from_db(db.is_correlated(self))

    def is_flapping(self, window: int = 1800, count: int = 2) -> bool:
        return db.is_flapping(self, window, count)

    def get_status_and_value(self) -> List[Tuple[str, Optional[str]]]:
        return [(h.status, h.value) for h in self.get_alert_history(self, page=1, page_size=10) if h.status]

    def _get_hist_info(self, action: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        h_loop = self.get_alert_history(alert=self)
        if not h_loop:
            return (None, None, None, None)
        current_status = h_loop[0].status
        current_value = h_loop[0].value
        if len(h_loop) == 1:
            return (current_status, current_value, None, None)
        if action == ChangeType.unack:
            find = ChangeType.ack
        elif action == ChangeType.unshelve:
            find = ChangeType.shelve
        else:
            find = None
        if find:
            for h, h_next in zip(h_loop, h_loop[1:]):
                if h.change_type == find:
                    return (current_status, current_value, h_next.status, h_next.timeout)
        return (current_status, current_value, h_loop[1].status, h_loop[1].timeout)

    def deduplicate(self, duplicate_of: 'Alert') -> 'Alert':
        now = datetime.utcnow()
        status, previous_value, previous_status, _ = self._get_hist_info()
        _, new_status = alarm_model.transition(alert=self, current_status=status, previous_status=previous_status)
        self.repeat = True
        self.last_receive_id = self.id
        self.last_receive_time = now
        if new_status != status:
            r = status_change_hook.send(duplicate_of, status=new_status, text=self.text)
            _, (_, new_status, text) = r[0]
            self.update_time = now
            history = History(
                id=self.id,
                event=self.event,
                severity=self.severity,
                status=new_status,
                value=self.value,
                text=text,
                change_type=ChangeType.status,
                update_time=self.create_time,
                user=g.login,
                timeout=self.timeout
            )
        elif current_app.config['HISTORY_ON_VALUE_CHANGE'] and self.value != previous_value:
            history = History(
                id=self.id,
                event=self.event,
                severity=self.severity,
                status=status,
                value=self.value,
                text=self.text,
                change_type=ChangeType.value,
                update_time=self.create_time,
                user=g.login,
                timeout=self.timeout
            )
        else:
            history = None
        self.status = new_status
        return Alert.from_db(db.dedup_alert(self, history))

    def update(self, correlate_with: 'Alert') -> 'Alert':
        now = datetime.utcnow()
        self.previous_severity = db.get_severity(self)
        self.trend_indication = alarm_model.trend(self.previous_severity, self.severity)
        status, _, previous_status, _ = self._get_hist_info()
        _, new_status = alarm_model.transition(alert=self, current_status=status, previous_status=previous_status)
        self.duplicate_count = 0
        self.repeat = False
        self.receive_time = now
        self.last_receive_id = self.id
        self.last_receive_time = now
        if new_status != status:
            r = status_change_hook.send(correlate_with, status=new_status, text=self.text)
            _, (_, new_status, text) = r[0]
            self.update_time = now
        else:
            text = self.text
        history = [History(
            id=self.id,
            event=self.event,
            severity=self.severity,
            status=new_status,
            value=self.value,
            text=text,
            change_type=ChangeType.severity,
            update_time=self.create_time,
            user=g.login,
            timeout=self.timeout
        )]
        self.status = new_status
        return Alert.from_db(db.correlate_alert(self, history))

    def create(self) -> 'Alert':
        now = datetime.utcnow()
        trend_indication = alarm_model.trend(alarm_model.DEFAULT_PREVIOUS_SEVERITY, self.severity)
        _, self.status = alarm_model.transition(alert=self)
        self.duplicate_count = 0
        self.repeat = False
        self.previous_severity = alarm_model.DEFAULT_PREVIOUS_SEVERITY
        self.trend_indication = trend_indication
        self.receive_time = now
        self.last_receive_id = self.id
        self.last_receive_time = now
        self.update_time = now
        self.history = [History(
            id=self.id,
            event=self.event,
            severity=self.severity,
            status=self.status,
            value=self.value,
            text=self.text,
            change_type=ChangeType.new,
            update_time=self.create_time,
            user=g.login,
            timeout=self.timeout
        )]
        return Alert.from_db(db.create_alert(self))

    @staticmethod
    def find_by_id(id: str, customers: Optional[List[str]] = None) -> Optional['Alert']:
        return Alert.from_db(db.get_alert(id, customers))

    def is_blackout(self) -> bool:
        """Does the alert create time fall within an existing blackout period?"""
        if not current_app.config['NOTIFICATION_BLACKOUT']:
            if self.severity in current_app.config['BLACKOUT_ACCEPT']:
                return False
        return db.is_blackout_period(self)

    @property
    def is_suppressed(self) -> bool:
        """Is the alert status 'blackout'?"""
        return alarm_model.is_suppressed(self)

    def set_status(self, status: str, text: str = '', timeout: Optional[int] = None) -> 'Alert':
        now = datetime.utcnow()
        timeout = timeout or current_app.config['ALERT_TIMEOUT']
        history = History(
            id=self.id,
            event=self.event,
            severity=self.severity,
            status=status,
            value=self.value,
            text=text,
            change_type=ChangeType.status,
            update_time=now,
            user=g.login,
            timeout=self.timeout
        )
        return Alert.from_db(db.set_status(self.id, status, timeout, update_time=now, history=history))

    def tag(self, tags: List[str]) -> bool:
        return db.tag_alert(self.id, tags)

    def