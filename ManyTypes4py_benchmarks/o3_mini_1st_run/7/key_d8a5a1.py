from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from strenum import StrEnum
from alerta.app import db, key_helper
from alerta.database.base import Query
from alerta.models.enums import Scope
from alerta.utils.format import DateTime
from alerta.utils.response import absolute_url

JSON = Dict[str, Any]


class ApiKeyStatus(StrEnum):
    Active = 'active'
    Expired = 'expired'


class ApiKey:

    def __init__(self, user: str, scopes: Optional[List[Scope]] = None, text: str = '', expire_time: Optional[datetime] = None, customer: Optional[str] = None, **kwargs: Any) -> None:
        self.id: str = kwargs.get('id') or str(uuid4())
        self.key: str = kwargs.get('key', None) or key_helper.generate()
        self.user: str = user
        self.scopes: List[Scope] = scopes or key_helper.user_default_scopes
        self.text: str = text
        self.expire_time: datetime = expire_time or datetime.utcnow() + timedelta(days=key_helper.api_key_expire_days)
        self.count: int = kwargs.get('count', 0)
        self.last_used_time: Optional[datetime] = kwargs.get('last_used_time', None)
        self.customer: Optional[str] = customer

    @property
    def type(self) -> str:
        return key_helper.scopes_to_type(self.scopes)

    @property
    def status(self) -> ApiKeyStatus:
        return ApiKeyStatus.Expired if datetime.utcnow() > self.expire_time else ApiKeyStatus.Active

    @classmethod
    def parse(cls, json: JSON) -> "ApiKey":
        if not isinstance(json.get('scopes', []), list):
            raise ValueError('scopes must be a list')
        api_key = ApiKey(
            id=json.get('id', None), 
            user=json.get('user', None), 
            scopes=[Scope(s) for s in json.get('scopes', [])], 
            text=json.get('text', None), 
            expire_time=DateTime.parse(json['expireTime']) if 'expireTime' in json else None, 
            customer=json.get('customer', None), 
            key=json.get('key')
        )
        if 'type' in json:
            api_key.scopes = key_helper.type_to_scopes(api_key.user, json['type'])
        return api_key

    @property
    def serialize(self) -> JSON:
        return {
            'id': self.id,
            'key': self.key,
            'status': self.status,
            'href': absolute_url('/key/' + self.key),
            'user': self.user,
            'scopes': self.scopes,
            'type': self.type,
            'text': self.text,
            'expireTime': self.expire_time,
            'count': self.count,
            'lastUsedTime': self.last_used_time,
            'customer': self.customer
        }

    def __repr__(self) -> str:
        return 'ApiKey(key={!r}, status={!r}, user={!r}, scopes={!r}, expireTime={!r}, customer={!r})'.format(
            self.key, self.status, self.user, self.scopes, self.expire_time, self.customer
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "ApiKey":
        return ApiKey(
            id=doc.get('id', None) or doc.get('_id'),
            key=doc.get('key', None) or doc.get('_id'),
            user=doc.get('user', None),
            scopes=[Scope(s) for s in doc.get('scopes', list())] or key_helper.type_to_scopes(doc.get('user', None), doc.get('type', None)) or list(),
            text=doc.get('text', None),
            expire_time=doc.get('expireTime', None),
            count=doc.get('count', None),
            last_used_time=doc.get('lastUsedTime', None),
            customer=doc.get('customer', None)
        )

    @classmethod
    def from_record(cls, rec: Any) -> "ApiKey":
        return ApiKey(
            id=rec.id,
            key=rec.key,
            user=rec.user,
            scopes=[Scope(s) for s in rec.scopes],
            text=rec.text,
            expire_time=rec.expire_time,
            count=rec.count,
            last_used_time=rec.last_used_time,
            customer=rec.customer
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional["ApiKey"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        return None

    def create(self) -> "ApiKey":
        """
        Create a new API key.
        """
        return ApiKey.from_db(db.create_key(self))

    @staticmethod
    def find_by_id(key: str, user: Optional[str] = None) -> Optional["ApiKey"]:
        """
        Get API key details.
        """
        return ApiKey.from_db(db.get_key(key, user))

    @staticmethod
    def find_all(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List["ApiKey"]:
        """
        List all API keys.
        """
        return [ApiKey.from_db(key) for key in db.get_keys(query, page, page_size) if ApiKey.from_db(key) is not None]

    @staticmethod
    def count(query: Optional[Query] = None) -> int:
        return db.get_keys_count(query)

    @staticmethod
    def find_by_user(user: str) -> List["ApiKey"]:
        """
        List API keys for a user.
        """
        return [ApiKey.from_db(key) for key in db.get_keys_by_user(user) if ApiKey.from_db(key) is not None]

    def update(self, **kwargs: Any) -> "ApiKey":
        kwargs['expireTime'] = DateTime.parse(kwargs['expireTime']) if 'expireTime' in kwargs else None
        return ApiKey.from_db(db.update_key(self.key, **kwargs))

    def delete(self) -> Any:
        """
        Delete an API key.
        """
        return db.delete_key(self.key)

    @staticmethod
    def verify_key(key: str) -> Optional["ApiKey"]:
        key_info: Optional[ApiKey] = ApiKey.from_db(db.get_key(key))
        if key_info and key_info.expire_time > datetime.utcnow():
            db.update_key_last_used(key)
            return key_info
        return None