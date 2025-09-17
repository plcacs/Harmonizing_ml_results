from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from flask import g
from alerta.app import db
from alerta.database.base import Query
from alerta.models.enums import NoteType
from alerta.utils.format import DateTime
from alerta.utils.response import absolute_url

JSON = Dict[str, Any]


class Note:

    def __init__(self, text: str, user: str, note_type: Union[str, NoteType], **kwargs: Any) -> None:
        self.id: str = kwargs.get('id') or str(uuid4())
        self.text: str = text
        self.user: str = user
        self.note_type: Union[str, NoteType] = note_type
        self.attributes: Dict[str, Any] = kwargs.get('attributes', None) or dict()
        self.create_time: datetime = kwargs['create_time'] if 'create_time' in kwargs else datetime.utcnow()
        self.update_time: Optional[datetime] = kwargs.get('update_time')
        self.alert: Optional[str] = kwargs.get('alert')
        self.customer: Optional[str] = kwargs.get('customer')

    @classmethod
    def parse(cls, json: JSON) -> "Note":
        return Note(
            id=json.get('id', None),
            text=json.get('status', None),
            user=json.get('status', None),
            attributes=json.get('attributes', dict()),
            note_type=json.get('type', None),
            create_time=DateTime.parse(json['createTime']) if 'createTime' in json else None,
            update_time=DateTime.parse(json['updateTime']) if 'updateTime' in json else None,
            alert=json.get('related', {}).get('alert'),
            customer=json.get('customer', None)
        )

    @property
    def serialize(self) -> JSON:
        note: JSON = {
            'id': self.id,
            'href': absolute_url('/note/' + self.id),
            'text': self.text,
            'user': self.user,
            'attributes': self.attributes,
            'type': self.note_type,
            'createTime': self.create_time,
            'updateTime': self.update_time,
            '_links': dict(),
            'customer': self.customer
        }
        if self.alert:
            note['related'] = {'alert': self.alert}
            note['_links'] = {'alert': absolute_url('/alert/' + self.alert)}
        return note

    def __repr__(self) -> str:
        return 'Note(id={!r}, text={!r}, user={!r}, type={!r}, customer={!r})'.format(
            self.id, self.text, self.user, self.note_type, self.customer)

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "Note":
        return Note(
            id=doc.get('id', None) or doc.get('_id'),
            text=doc.get('text', None),
            user=doc.get('user', None),
            attributes=doc.get('attributes', dict()),
            note_type=doc.get('type', None),
            create_time=doc.get('createTime'),
            update_time=doc.get('updateTime'),
            alert=doc.get('alert'),
            customer=doc.get('customer')
        )

    @classmethod
    def from_record(cls, rec: Any) -> "Note":
        return Note(
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
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional["Note"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        return None

    def create(self) -> "Note":
        return Note.from_db(db.create_note(self))  # type: ignore

    @staticmethod
    def from_alert(alert: Any, text: str) -> "Note":
        note = Note(
            text=text,
            user=g.login,  # type: ignore
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
    def find_by_id(id: str) -> "Note":
        return Note.from_db(db.get_note(id))  # type: ignore

    @staticmethod
    def find_all(query: Optional[Query] = None) -> List["Note"]:
        return [Note.from_db(note) for note in db.get_notes(query)]  # type: ignore

    def update(self, **kwargs: Any) -> "Note":
        return Note.from_db(db.update_note(self.id, **kwargs))  # type: ignore

    def delete(self) -> Any:
        return db.delete_note(self.id)

    @staticmethod
    def delete_by_id(id: str) -> Any:
        return db.delete_note(id)