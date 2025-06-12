import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from uuid import uuid4, UUID
from flask import current_app
from strenum import StrEnum
from alerta.app import db
from alerta.database.base import Query
from alerta.utils.format import DateTime
from alerta.utils.response import absolute_url
JSON = Dict[str, Any]

class HeartbeatStatus(StrEnum):
    OK = 'ok'
    Slow = 'slow'
    Expired = 'expired'

class Heartbeat:

    def __init__(
        self,
        origin: Optional[str] = None,
        tags: Optional[List[str]] = None,
        create_time: Optional[datetime] = None,
        timeout: Optional[int] = None,
        customer: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        if any(['.' in key for key in kwargs.get('attributes', dict()).keys()]) or any(['$' in key for key in kwargs.get('attributes', dict()).keys()]):
            raise ValueError('Attribute keys must not contain "." or "$"')
        timeout = timeout if timeout is not None else current_app.config['HEARTBEAT_TIMEOUT']
        max_latency = current_app.config['HEARTBEAT_MAX_LATENCY']
        try:
            timeout = int(timeout)
        except ValueError:
            raise ValueError(f"Could not convert 'timeout' value of '{timeout}' to an integer")
        if timeout < 0:
            raise ValueError(f"Invalid negative 'timeout' value ({timeout})")
        try:
            max_latency = int(max_latency)
        except ValueError:
            raise ValueError(f"Could not convert 'max_latency' value of '{timeout}' to an integer")
        if timeout < 0:
            raise ValueError(f"Invalid negative 'max_latency' value ({timeout})")
        self.id: str = kwargs.get('id') or str(uuid4())
        self.origin: str = origin or f'{os.path.basename(sys.argv[0])}/{platform.uname()[1]}'
        self.tags: List[str] = tags or list()
        self.attributes: Dict[str, Any] = kwargs.get('attributes', None) or dict()
        self.event_type: str = kwargs.get('event_type', kwargs.get('type', None)) or 'Heartbeat'
        self.create_time: datetime = create_time or datetime.utcnow()
        self.timeout: int = timeout
        self.max_latency: int = max_latency
        self.receive_time: datetime = kwargs.get('receive_time', None) or datetime.utcnow()
        self.latency: int = int((self.receive_time - self.create_time).total_seconds() * 1000)
        self.since: datetime = datetime.utcnow() - self.receive_time
        self.customer: Optional[str] = customer

    @property
    def status(self) -> HeartbeatStatus:
        if self.since.total_seconds() > self.timeout:
            return HeartbeatStatus.Expired
        elif self.latency > self.max_latency:
            return HeartbeatStatus.Slow
        return HeartbeatStatus.OK

    @classmethod
    def parse(cls, json: JSON) -> 'Heartbeat':
        if not isinstance(json.get('tags', []), list):
            raise ValueError('tags must be a list')
        if not isinstance(json.get('timeout') if json.get('timeout', None) is not None else 0, int):
            raise ValueError('timeout must be an integer')
        if not isinstance(json.get('attributes', {}), dict):
            raise ValueError('attributes must be a JSON object')
        if json.get('customer', None) == '':
            raise ValueError('customer must not be an empty string')
        return Heartbeat(
            id=json.get('id', None),
            origin=json.get('origin', None),
            tags=json.get('tags', list()),
            attributes=json.get('attributes', dict()),
            create_time=DateTime.parse(json['createTime']) if 'createTime' in json else None,
            timeout=json.get('timeout', None),
            customer=json.get('customer', None)
        )

    @property
    def serialize(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'href': absolute_url('/heartbeat/' + self.id),
            'origin': self.origin,
            'tags': self.tags,
            'attributes': self.attributes,
            'type': self.event_type,
            'createTime': self.create_time,
            'timeout': self.timeout,
            'maxLatency': self.max_latency,
            'receiveTime': self.receive_time,
            'customer': self.customer,
            'latency': self.latency,
            'since': self.since,
            'status': self.status
        }

    def __repr__(self) -> str:
        return 'Heartbeat(id={!r}, origin={!r}, create_time={!r}, timeout={!r}, customer={!r})'.format(
            self.id, self.origin, self.create_time, self.timeout, self.customer)

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Heartbeat':
        return Heartbeat(
            id=doc.get('id', None) or doc.get('_id'),
            origin=doc.get('origin', None),
            tags=doc.get('tags', list()),
            attributes=doc.get('attributes', dict()),
            event_type=doc.get('type', None),
            create_time=doc.get('createTime', None),
            timeout=doc.get('timeout', None),
            receive_time=doc.get('receiveTime', None),
            latency=doc.get('latency', None),
            since=doc.get('since', None),
            customer=doc.get('customer', None)
        )

    @classmethod
    def from_record(cls, rec: Any) -> 'Heartbeat':
        return Heartbeat(
            id=rec.id,
            origin=rec.origin,
            tags=rec.tags,
            attributes=dict(getattr(rec, 'attributes') or ()),
            event_type=rec.type,
            create_time=rec.create_time,
            timeout=rec.timeout,
            receive_time=rec.receive_time,
            latency=rec.latency,
            since=rec.since,
            customer=rec.customer
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> 'Heartbeat':
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        raise ValueError("Invalid database record type")

    def create(self) -> 'Heartbeat':
        return Heartbeat.from_db(db.upsert_heartbeat(self))

    @staticmethod
    def find_by_id(id: str, customers: Optional[List[str]] = None) -> Optional['Heartbeat']:
        return Heartbeat.from_db(db.get_heartbeat(id, customers))

    @staticmethod
    def find_all(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['Heartbeat']:
        return [Heartbeat.from_db(heartbeat) for heartbeat in db.get_heartbeats(query, page, page_size)]

    @staticmethod
    def find_all_by_status(status: Optional[HeartbeatStatus] = None, query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List['Heartbeat']:
        return [Heartbeat.from_db(heartbeat) for heartbeat in db.get_heartbeats_by_status(status, query, page, page_size)]

    @staticmethod
    def count(query: Optional[Query] = None) -> int:
        return db.get_heartbeats_count(query)

    def delete(self) -> bool:
        return db.delete_heartbeat(self.id)
