from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from typing import Optional
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4
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


class Alert:

    def __init__(self, resource: str, event: str, **kwargs: Any) -> None:
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
        timeout: Union[int, str, None] = kwargs.get('timeout') if kwargs.get('timeout') is not None else current_app.config['ALERT_TIMEOUT']
        try:
            timeout = int(timeout)  # type: ignore[arg-type]
        except ValueError:
            raise ValueError(f"Could not convert 'timeout' value of '{timeout}' to an integer")
        if timeout < 0:
            raise ValueError(f"Invalid negative 'timeout' value ({timeout})")
        self.id: str = kwargs.get('id') or str(uuid4())
        self.resource: str = resource
        self.event: str = event
        self.environment: str = kwargs.get('environment', None) or ''
        self.severity: str = kwargs.get('severity', None) or alarm_model.DEFAULT_NORMAL_SEVERITY
        self.correlate: List[str] = kwargs.get('correlate', None) or list()
        if self.correlate and event not in self.correlate:
            self.correlate.append(event)
        self.status: str = kwargs.get('status', None) or alarm_model.DEFAULT_STATUS
        self.service: List[str] = kwargs.get('service', None) or list()
        self.group: str = kwargs.get('group', None) or 'Misc'
        self.value: Optional[str] = kwargs.get('value', None)
        self.text: str = kwargs.get('text', None) or ''
        self.tags: List[str] = kwargs.get('tags', None) or list()
        self.attributes: Dict[str, Any] = kwargs.get('attributes', None) or dict()
        self.origin: str = kwargs.get('origin', None) or f'{os.path.basename(sys.argv[0])}/{platform.uname()[1]}'
        self.event_type: str = kwargs.get('event_type', kwargs.get('type', None)) or 'exceptionAlert'
        self.create_time: datetime = kwargs.get('create_time', None) or datetime.utcnow()
        self.timeout: int = int(timeout)
        self.raw_data: Optional[str] = kwargs.get('raw_data', None)
        self.customer: Optional[str] = kwargs.get('customer', None)
        self.duplicate_count: Optional[int] = kwargs.get('duplicate_count', None)
        self.repeat: Optional[bool] = kwargs.get('repeat', None)
        self.previous_severity: Optional[str] = kwargs.get('previous_severity', None)
        self.trend_indication: Optional[str] = kwargs.get('trend_indication', None)
        self.receive_time: datetime = kwargs.get('receive_time', None) or datetime.utcnow()
        self.last_receive_id: Optional[str] = kwargs.get('last_receive_id', None)
        self.last_receive_time: Optional[datetime] = kwargs.get('last_receive_time', None)
        self.update_time: Optional[datetime] = kwargs.get('update_time', None)
        self.history: List[History] = kwargs.get('history', None) or list()

    @classmethod
    def parse(cls, json: JSON) -> Alert:
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
        return Alert(
            id=json.get('id', None),
            resource=json.get('resource', None),
            event=json.get('event', None),
            environment=json.get('environment', None),
            severity=json.get('severity', None),
            correlate=json.get('correlate', list()),
            status=json.get('status', None),
            service=json.get('service', list()),
            group=json.get('group', None),
            value=json.get('value', None),
            text=json.get('text', None),
            tags=json.get('tags', list()),
            attributes=json.get('attributes', dict()),
            origin=json.get('origin', None),
            event_type=json.get('type', None),
            create_time=DateTime.parse(json['createTime']) if 'createTime' in json else None,
            timeout=json.get('timeout', None),
            raw_data=json.get('rawData', None),
            customer=json.get('customer', None)
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
        return 'Alert(id={!r}, environment={!r}, resource={!r}, event={!r}, severity={!r}, status={!r}, customer={!r})'.format(self.id, self.environment, self.resource, self.event, self.severity, self.status, self.customer)

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> Alert:
        return Alert(
            id=doc.get('id', None) or doc.get('_id'),
            resource=doc.get('resource', None),
            event=doc.get('event', None),
            environment=doc.get('environment', None),
            severity=doc.get('severity', None),
            correlate=doc.get('correlate', list()),
            status=doc.get('status', None),
            service=doc.get('service', list()),
            group=doc.get('group', None),
            value=doc.get('value', None),
            text=doc.get('text', None),
            tags=doc.get('tags', list()),
            attributes=doc.get('attributes', dict()),
            origin=doc.get('origin', None),
            event_type=doc.get('type', None),
            create_time=doc.get('createTime', None),
            timeout=doc.get('timeout', None),
            raw_data=doc.get('rawData', None),
            customer=doc.get('customer', None),
            duplicate_count=doc.get('duplicateCount', None),
            repeat=doc.get('repeat', None),
            previous_severity=doc.get('previousSeverity', None),
            trend_indication=doc.get('trendIndication', None),
            receive_time=doc.get('receiveTime', None),
            last_receive_id=doc.get('lastReceiveId', None),
            last_receive_time=doc.get('lastReceiveTime', None),
            update_time=doc.get('updateTime', None),
            history=[History.from_db(h) for h in doc.get('history', list())]
        )

    @classmethod
    def from_record(cls, rec: Any) -> Alert:
        return Alert(
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
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...], None]) -> Optional[Alert]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        return None

    def is_duplicate(self) -> Optional[Alert]:
        """Return duplicate alert or None"""
        return Alert.from_db(db.is_duplicate(self))

    def is_correlated(self) -> Optional[Alert]:
        """Return correlated alert or None"""
        return Alert.from_db(db.is_correlated(self))

    def is_flapping(self, window: int = 1800, count: int = 2) -> bool:
        return db.is_flapping(self, window, count)

    def get_status_and_value(self) -> List[Tuple[str, Any]]:
        return [(h.status, h.value) for h in self.get_alert_history(self, page=1, page_size=10) if h.status]

    def _get_hist_info(self, action: Optional[Union[str, ChangeType]] = None) -> Tuple[Optional[str], Optional[Any], Optional[str], Optional[int]]:
        h_loop = self.get_alert_history(alert=self)
        if not h_loop:
            return (None, None, None, None)
        current_status: Optional[str] = h_loop[0].status
        current_value: Optional[Any] = h_loop[0].value
        if len(h_loop) == 1:
            return (current_status, current_value, None, None)
        if action == ChangeType.unack:
            find: Optional[ChangeType] = ChangeType.ack
        elif action == ChangeType.unshelve:
            find = ChangeType.shelve
        else:
            find = None
        if find:
            for h, h_next in zip(h_loop, h_loop[1:]):
                if h.change_type == find:
                    return (current_status, current_value, h_next.status, h_next.timeout)
        return (current_status, current_value, h_loop[1].status, h_loop[1].timeout)

    def deduplicate(self, duplicate_of: Alert) -> Optional[Alert]:
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
            history: Optional[History] = History(id=self.id, event=self.event, severity=self.severity, status=new_status, value=self.value, text=text, change_type=ChangeType.status, update_time=self.create_time, user=g.login, timeout=self.timeout)
        elif current_app.config['HISTORY_ON_VALUE_CHANGE'] and self.value != previous_value:
            history = History(id=self.id, event=self.event, severity=self.severity, status=status, value=self.value, text=self.text, change_type=ChangeType.value, update_time=self.create_time, user=g.login, timeout=self.timeout)
        else:
            history = None
        self.status = new_status
        return Alert.from_db(db.dedup_alert(self, history))

    def update(self, correlate_with: Alert) -> Optional[Alert]:
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
        history = [History(id=self.id, event=self.event, severity=self.severity, status=new_status, value=self.value, text=text, change_type=ChangeType.severity, update_time=self.create_time, user=g.login, timeout=self.timeout)]
        self.status = new_status
        return Alert.from_db(db.correlate_alert(self, history))

    def create(self) -> Optional[Alert]:
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
        self.history = [History(id=self.id, event=self.event, severity=self.severity, status=self.status, value=self.value, text=self.text, change_type=ChangeType.new, update_time=self.create_time, user=g.login, timeout=self.timeout)]
        return Alert.from_db(db.create_alert(self))

    @staticmethod
    def find_by_id(id: str, customers: Optional[List[str]] = None) -> Optional[Alert]:
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

    def set_status(self, status: str, text: str = '', timeout: Optional[int] = None) -> Optional[Alert]:
        now = datetime.utcnow()
        timeout_val: int = timeout or current_app.config['ALERT_TIMEOUT']
        history = History(id=self.id, event=self.event, severity=self.severity, status=status, value=self.value, text=text, change_type=ChangeType.status, update_time=now, user=g.login, timeout=self.timeout)
        return Alert.from_db(db.set_status(self.id, status, timeout_val, update_time=now, history=history))

    def tag(self, tags: List[str]) -> Any:
        return db.tag_alert(self.id, tags)

    def untag(self, tags: List[str]) -> Any:
        return db.untag_alert(self.id, tags)

    def update_tags(self, tags: List[str]) -> Any:
        return db.update_tags(self.id, list(set(tags)))

    def update_attributes(self, attributes: Dict[str, Any]) -> Any:
        return db.update_attributes(self.id, self.attributes, attributes)

    def delete(self) -> Any:
        return db.delete_alert(self.id)

    @staticmethod
    def tag_find_all(query: Optional[Query], tags: List[str]) -> Any:
        return db.tag_alerts(query, tags)

    @staticmethod
    def untag_find_all(query: Optional[Query], tags: List[str]) -> Any:
        return db.untag_alerts(query, tags)

    @staticmethod
    def update_attributes_find_all(query: Optional[Query], attributes: Dict[str, Any]) -> Any:
        return db.update_attributes_by_query(query, attributes)

    @staticmethod
    def delete_find_all(query: Optional[Query] = None) -> Any:
        return db.delete_alerts(query)

    @staticmethod
    def find_all(query: Optional[Query] = None, raw_data: bool = False, history: bool = False, page: int = 1, page_size: int = 1000) -> List[Alert]:
        return [Alert.from_db(alert) for alert in db.get_alerts(query, raw_data, history, page, page_size)]  # type: ignore[list-item]

    @staticmethod
    def get_alert_history(alert: Alert, page: int = 1, page_size: int = 100) -> List[RichHistory]:
        return [RichHistory.from_db(hist) for hist in db.get_alert_history(alert, page, page_size)]

    @staticmethod
    def get_history(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List[RichHistory]:
        return [RichHistory.from_db(hist) for hist in db.get_history(query, page, page_size)]

    @staticmethod
    def get_count(query: Optional[Query] = None) -> int:
        return db.get_count(query)

    @staticmethod
    def get_counts_by_severity(query: Optional[Query] = None) -> Any:
        return db.get_counts_by_severity(query)

    @staticmethod
    def get_counts_by_status(query: Optional[Query] = None) -> Any:
        return db.get_counts_by_status(query)

    @staticmethod
    def get_top10_count(query: Optional[Query] = None) -> Any:
        return Alert.get_topn_count(query, topn=10)

    @staticmethod
    def get_topn_count(query: Optional[Query] = None, topn: int = 10) -> Any:
        return db.get_topn_count(query, topn=topn)

    @staticmethod
    def get_top10_flapping(query: Optional[Query] = None) -> Any:
        return Alert.get_topn_flapping(topn=10)

    @staticmethod
    def get_topn_flapping(query: Optional[Query] = None, topn: int = 10) -> Any:
        return db.get_topn_flapping(query, topn=topn)

    @staticmethod
    def get_top10_standing(query: Optional[Query] = None) -> Any:
        return Alert.get_topn_standing(topn=10)

    @staticmethod
    def get_topn_standing(query: Optional[Query] = None, topn: int = 10) -> Any:
        return db.get_topn_standing(query, topn=topn)

    @staticmethod
    def get_environments(query: Optional[Query] = None) -> Any:
        return db.get_environments(query)

    @staticmethod
    def get_services(query: Optional[Query] = None) -> Any:
        return db.get_services(query)

    @staticmethod
    def get_groups(query: Optional[Query] = None) -> Any:
        return db.get_alert_groups(query)

    @staticmethod
    def get_tags(query: Optional[Query] = None) -> Any:
        return db.get_alert_tags(query)

    def add_note(self, text: str) -> Note:
        note = Note.from_alert(self, text)
        history = History(id=note.id, event=self.event, severity=self.severity, status=self.status, value=self.value, text=text, change_type=ChangeType.note, update_time=datetime.utcnow(), user=g.login)
        db.add_history(self.id, history)
        return note

    def get_alert_notes(self, page: int = 1, page_size: int = 100) -> List[Note]:
        notes = db.get_alert_notes(self.id, page, page_size)
        return [Note.from_db(note) for note in notes]

    def delete_note(self, note_id: str) -> Any:
        history = History(id=note_id, event=self.event, severity=self.severity, status=self.status, value=self.value, text='note dismissed', change_type=ChangeType.dismiss, update_time=datetime.utcnow(), user=g.login)
        db.add_history(self.id, history)
        return Note.delete_by_id(note_id)

    @staticmethod
    def housekeeping(expired_threshold: int, info_threshold: int) -> Tuple[List[Optional[Alert]], List[Optional[Alert]], List[Optional[Alert]]]:
        return (
            [Alert.from_db(alert) for alert in db.get_expired(expired_threshold, info_threshold)],
            [Alert.from_db(alert) for alert in db.get_unshelve()],
            [Alert.from_db(alert) for alert in db.get_unack()]
        )

    def from_status(self, status: str, text: str = '', timeout: Optional[int] = None) -> Optional[Alert]:
        now = datetime.utcnow()
        self.timeout = timeout or current_app.config['ALERT_TIMEOUT']
        history = [History(id=self.id, event=self.event, severity=self.severity, status=status, value=self.value, text=text, change_type=ChangeType.status, update_time=now, user=g.login, timeout=self.timeout)]
        return Alert.from_db(db.set_alert(id=self.id, severity=self.severity, status=status, tags=self.tags, attributes=self.attributes, timeout=timeout, previous_severity=self.previous_severity, update_time=now, history=history))

    def from_action(self, action: Union[str, ChangeType], text: str = '', timeout: Optional[int] = None) -> Optional[Alert]:
        now = datetime.utcnow()
        status, _, previous_status, previous_timeout = self._get_hist_info(action)
        if action in [ChangeType.unack, ChangeType.unshelve, ChangeType.timeout]:
            timeout = timeout or previous_timeout
        if action in [ChangeType.ack, ChangeType.unack]:
            timeout = timeout or current_app.config['ACK_TIMEOUT']
        elif action in [ChangeType.shelve, ChangeType.unshelve]:
            timeout = timeout or current_app.config['SHELVE_TIMEOUT']
        else:
            timeout = timeout or self.timeout or current_app.config['ALERT_TIMEOUT']
        new_severity, new_status = alarm_model.transition(alert=self, current_status=status, previous_status=previous_status, action=action)
        r = status_change_hook.send(self, status=new_status, text=text)
        _, (_, new_status, text) = r[0]
        try:
            change_type = ChangeType(action)  # type: ignore[arg-type]
        except ValueError:
            change_type = ChangeType.action
        history = [History(id=self.id, event=self.event, severity=new_severity, status=new_status, value=self.value, text=text, change_type=change_type, update_time=now, user=g.login, timeout=timeout)]
        return Alert.from_db(db.set_alert(id=self.id, severity=new_severity, status=new_status, tags=self.tags, attributes=self.attributes, timeout=self.timeout, previous_severity=self.severity if new_severity != self.severity else self.previous_severity, update_time=now, history=history))

    def from_expired(self, text: str = '', timeout: Optional[int] = None) -> Optional[Alert]:
        return self.from_action(action='expired', text=text, timeout=timeout)

    def from_timeout(self, text: str = '', timeout: Optional[int] = None) -> Optional[Alert]:
        return self.from_action(action='timeout', text=text, timeout=timeout)