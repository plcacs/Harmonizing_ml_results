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

    def __init__(self, user: Union[str, contests.models.User, zerver.models.UserProfile], scopes: Union[str, None, list[str], list["ResourceTypeId"]], text: typing.Text='', expire_time: Union[None, datetime.datetime, datetime.date]=None, customer: Union[None, str, bool, datetime.datetime]=None, **kwargs) -> None:
        self.id = kwargs.get('id') or str(uuid4())
        self.key = kwargs.get('key', None) or key_helper.generate()
        self.user = user
        self.scopes = scopes or key_helper.user_default_scopes
        self.text = text
        self.expire_time = expire_time or datetime.utcnow() + timedelta(days=key_helper.api_key_expire_days)
        self.count = kwargs.get('count', 0)
        self.last_used_time = kwargs.get('last_used_time', None)
        self.customer = customer

    @property
    def type(self):
        return key_helper.scopes_to_type(self.scopes)

    @property
    def status(self):
        return ApiKeyStatus.Expired if datetime.utcnow() > self.expire_time else ApiKeyStatus.Active

    @classmethod
    def parse(cls: dict, json: Any) -> ApiKey:
        if not isinstance(json.get('scopes', []), list):
            raise ValueError('scopes must be a list')
        api_key = ApiKey(id=json.get('id', None), user=json.get('user', None), scopes=[Scope(s) for s in json.get('scopes', [])], text=json.get('text', None), expire_time=DateTime.parse(json['expireTime']) if 'expireTime' in json else None, customer=json.get('customer', None), key=json.get('key'))
        if 'type' in json:
            api_key.scopes = key_helper.type_to_scopes(api_key.user, json['type'])
        return api_key

    @property
    def serialize(self) -> dict[typing.Text, ]:
        return {'id': self.id, 'key': self.key, 'status': self.status, 'href': absolute_url('/key/' + self.key), 'user': self.user, 'scopes': self.scopes, 'type': self.type, 'text': self.text, 'expireTime': self.expire_time, 'count': self.count, 'lastUsedTime': self.last_used_time, 'customer': self.customer}

    def __repr__(self) -> str:
        return 'ApiKey(key={!r}, status={!r}, user={!r}, scopes={!r}, expireTime={!r}, customer={!r})'.format(self.key, self.status, self.user, self.scopes, self.expire_time, self.customer)

    @classmethod
    def from_document(cls: Union[dict, str, dict[str, str]], doc: Union[dict[str, typing.Any], dict]) -> ApiKey:
        return ApiKey(id=doc.get('id', None) or doc.get('_id'), key=doc.get('key', None) or doc.get('_id'), user=doc.get('user', None), scopes=[Scope(s) for s in doc.get('scopes', list())] or key_helper.type_to_scopes(doc.get('user', None), doc.get('type', None)) or list(), text=doc.get('text', None), expire_time=doc.get('expireTime', None), count=doc.get('count', None), last_used_time=doc.get('lastUsedTime', None), customer=doc.get('customer', None))

    @classmethod
    def from_record(cls: Union[dict, list[dict], str, None], rec: Union[dict, list, dict[str, typing.Any]]) -> ApiKey:
        return ApiKey(id=rec.id, key=rec.key, user=rec.user, scopes=[Scope(s) for s in rec.scopes], text=rec.text, expire_time=rec.expire_time, count=rec.count, last_used_time=rec.last_used_time, customer=rec.customer)

    @classmethod
    def from_db(cls: Any, r: Union[dict, tuple, str, None]):
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)

    def create(self):
        """
        Create a new API key.
        """
        return ApiKey.from_db(db.create_key(self))

    @staticmethod
    def find_by_id(key: Union[core.models.UserKey, str, int], user: Union[None, core.models.UserKey, str, int]=None):
        """
        Get API key details.
        """
        return ApiKey.from_db(db.get_key(key, user))

    @staticmethod
    def find_all(query: Union[None, int, str, alerta.database.base.Query]=None, page: int=1, page_size: int=1000) -> list:
        """
        List all API keys.
        """
        return [ApiKey.from_db(key) for key in db.get_keys(query, page, page_size)]

    @staticmethod
    def count(query: Union[None, alerta.database.base.Query, str]=None):
        return db.get_keys_count(query)

    @staticmethod
    def find_by_user(user: Union[str, users.models.JustfixUser, core.models.User]) -> list:
        """
        List API keys for a user.
        """
        return [ApiKey.from_db(key) for key in db.get_keys_by_user(user)]

    def update(self, **kwargs):
        kwargs['expireTime'] = DateTime.parse(kwargs['expireTime']) if 'expireTime' in kwargs else None
        return ApiKey.from_db(db.update_key(self.key, **kwargs))

    def delete(self):
        """
        Delete an API key.
        """
        return db.delete_key(self.key)

    @staticmethod
    def verify_key(key: Union[str, dict]) -> None:
        key_info = ApiKey.from_db(db.get_key(key))
        if key_info and key_info.expire_time > datetime.utcnow():
            db.update_key_last_used(key)
            return key_info
        return None