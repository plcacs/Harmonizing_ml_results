from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from alerta.app import db
from alerta.database.base import Query
from alerta.utils.response import absolute_url

JSON = Dict[str, Any]

class GroupUser:

    def __init__(self, id: str, login: str, name: str, status: str) -> None:
        self.id: str = id
        self.login: str = login
        self.name: str = name
        self.status: str = status

    @property
    def serialize(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'href': absolute_url('/user/' + self.id),
            'login': self.login,
            'name': self.name,
            'status': self.status
        }

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'GroupUser':
        return cls(
            id=doc.get('id', ''),
            name=doc.get('name', ''),
            login=doc.get('login', doc.get('email', '')),
            status=doc.get('status', '')
        )

    @classmethod
    def from_record(cls, rec: Any) -> 'GroupUser':
        return cls(
            id=rec.id,
            login=rec.login or rec.email,
            name=rec.name,
            status=rec.status
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> 'GroupUser':
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            raise TypeError("Unsupported type for from_db")

class GroupUsers:

    def __init__(self, id: str, users: List[GroupUser]) -> None:
        self.id: str = id
        self.users: List[GroupUser] = users

    @staticmethod
    def find_by_id(id: str) -> List[GroupUser]:
        return [GroupUser.from_db(user) for user in db.get_group_users(id)]

class Group:
    """
    Group model.
    """

    def __init__(self, name: str, text: str, **kwargs: Any) -> None:
        if not name:
            raise ValueError('Missing mandatory value for name')
        self.id: str = kwargs.get('id') or str(uuid4())
        self.name: str = name
        self.text: str = text or ''
        self.count: Optional[int] = kwargs.get('count')

    @classmethod
    def parse(cls, json: JSON) -> 'Group':
        return cls(
            id=json.get('id'),
            name=json.get('name', ''),
            text=json.get('text', '')
        )

    @property
    def serialize(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'href': absolute_url('/group/' + self.id),
            'name': self.name,
            'text': self.text,
            'count': self.count
        }

    def __repr__(self) -> str:
        return f'Group(id={self.id!r}, name={self.name!r}, text={self.text!r}, count={self.count!r})'

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> 'Group':
        return cls(
            id=doc.get('id') or doc.get('_id', ''),
            name=doc.get('name', ''),
            text=doc.get('text', ''),
            count=len(doc.get('users', []))
        )

    @classmethod
    def from_record(cls, rec: Any) -> 'Group':
        return cls(
            id=rec.id,
            name=rec.name,
            text=rec.text,
            count=rec.count
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> 'Group':
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            raise TypeError("Unsupported type for from_db")

    def create(self) -> 'Group':
        return Group.from_db(db.create_group(self))

    @staticmethod
    def find_by_id(id: str) -> Optional['Group']:
        group = db.get_group(id)
        if group:
            return Group.from_db(group)
        return None

    @staticmethod
    def find_all(query: Optional[Dict[str, Any]] = None, page: int = 1, page_size: int = 1000) -> List['Group']:
        return [Group.from_db(group) for group in db.get_groups(query, page, page_size)]

    @staticmethod
    def count(query: Optional[Dict[str, Any]] = None) -> int:
        return db.get_groups_count(query)

    def update(self, **kwargs: Any) -> 'Group':
        return Group.from_db(db.update_group(self.id, **kwargs))

    def add_user(self, user_id: str) -> Any:
        return db.add_user_to_group(group=self.id, user=user_id)

    def remove_user(self, user_id: str) -> Any:
        return db.remove_user_from_group(group=self.id, user=user_id)

    def delete(self) -> Any:
        return db.delete_group(self.id)
