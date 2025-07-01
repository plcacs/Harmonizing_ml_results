from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar
from uuid import uuid4
from flask import g
from alerta.app import db
from alerta.database.base import Query
from alerta.models.enums import NoteType
from alerta.utils.format import DateTime
from alerta.utils.response import absolute_url

JSON = Dict[str, Any]
T = TypeVar('T', bound='Note')


class Note:

    def __init__(self, text: str, user: str, note_type: NoteType, **kwargs: Any) -> None:
        self.id: str = kwargs.get('id') or str(uuid4())
        self.text: str = text
        self.user: str = user
        self.note_type: NoteType = note_type
        self.attributes: Dict[str, Any] = kwargs.get('attributes', dict())
        self.create_time: datetime = kwargs['create_time'] if 'create_time' in kwargs else datetime.utcnow()
        self.update_time: Optional[datetime] = kwargs.get('update_time')
        self.alert: Optional[str] = kwargs.get('alert')
        self.customer: Optional[str] = kwargs.get('customer')

    @classmethod
    def parse(cls: Type[T], json: JSON) -> T:
        return cls(
            id=json.get('id'),
            text=json.get('status'),
            user=json.get('status'),
            attributes=json.get('attributes', dict()),
            note_type=json.get('type'),
            create_time=DateTime.parse(json['createTime']) if 'createTime' in json else None,
            update_time=DateTime.parse(json['updateTime']) if 'updateTime' in json else None,
            alert=json.get('related', {}).get('alert'),
            customer=json.get('customer')
        )

    @property
    def serialize(self) -> Dict[str, Any]:
        note: Dict[str, Any] = {
            'id': self.id,
            'href': absolute_url('/note/' + self.id),
            'text': self.text,
            'user': self.user,
            'attributes': self.attributes,
            'type': self.note_type,
            'createTime': self.create_time,
            'updateTime': self.update_time,
            '_links': {},
            'customer': self.customer
        }
        if self.alert:
            note['related'] = {'alert': self.alert}
            note['_links'] = {'alert': absolute_url('/alert/' + self.alert)}
        return note

    def __repr__(self) -> str:
        return f"Note(id={self.id!r}, text={self.text!r}, user={self.user!r}, type={self.note_type!r}, customer={self.customer!r})"

    @classmethod
    def from_document(cls: Type[T], doc: Dict[str, Any]) -> T:
        return cls(
            id=doc.get('id') or doc.get('_id'),
            text=doc.get('text'),
            user=doc.get('user'),
            attributes=doc.get('attributes', dict()),
            note_type=doc.get('type'),
            create_time=doc.get('createTime'),
            update_time=doc.get('updateTime'),
            alert=doc.get('alert'),
            customer=doc.get('customer')
        )

    @classmethod
    def from_record(cls: Type[T], rec: Any) -> T:
        return cls(
            id=rec.id,
            text=rec.text,
            user=rec.user,
            attributes=dict(rec.attributes),
            note_type=rec.type,
            create_time=rec.create_time,
            update_time=rec.update_time,
            alert=rec.alert,
            customer=rec.customer
        )

    @classmethod
    def from_db(cls: Type[T], r: Union[Dict[str, Any], Any]) -> Optional[T]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple) or hasattr(r, 'id'):
            return cls.from_record(r)
        return None

    def create(self) -> Optional['Note']:
        return Note.from_db(db.create_note(self))

    @staticmethod
    def from_alert(alert: Any, text: str) -> Optional['Note']:
        note = Note(
            text=text,
            user=g.login,
            note_type=NoteType.alert,
            attributes=dict(
                resource=alert.resource,
                event=alert.event,
                environment=alert.environment,
                severity=alert.severity,
                status=alert.status
            ),
            alert=alert.id,
            customer=alert.customer
        )
        return note.create()

    @staticmethod
    def find_by_id(id: str) -> Optional['Note']:
        return Note.from_db(db.get_note(id))

    @staticmethod
    def find_all(query: Optional[Dict[str, Any]] = None) -> List['Note']:
        return [Note.from_db(note) for note in db.get_notes(query)]

    def update(self, **kwargs: Any) -> Optional['Note']:
        return Note.from_db(db.update_note(self.id, **kwargs))

    def delete(self) -> None:
        db.delete_note(self.id)

    @staticmethod
    def delete_by_id(id: str) -> None:
        db.delete_note(id)
