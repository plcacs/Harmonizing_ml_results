from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from flask import current_app
from strenum import StrEnum

from alerta.app import db
from alerta.auth import utils
from alerta.database.base import Query
from alerta.models.group import Group
from alerta.utils.response import absolute_url

JSON = Dict[str, Any]


class UserStatus(StrEnum):
    Active = 'active'
    Inactive = 'inactive'
    Unknown = 'unknown'


class User:
    """
    User model for all auth providers.
    """

    def __init__(
        self,
        name: Optional[str],
        login: str,
        password: str,
        email: Optional[str],
        roles: List[str],
        text: Optional[str],
        **kwargs: Any
    ) -> None:
        if not login:
            raise ValueError('Missing mandatory value for "login"')
        self.id: str = kwargs.get('id') or str(uuid4())
        self.name: str = name or ''
        self.login: str = login
        self.password: str = password
        self.email: Optional[str] = email
        self.status: UserStatus = kwargs.get('status', None) or UserStatus.Active
        self.roles: List[str] = current_app.config['ADMIN_ROLES'] if (
            self.email and self.email in current_app.config['ADMIN_USERS']
        ) else roles
        self.attributes: Dict[str, Any] = kwargs.get('attributes', None) or dict()
        self.create_time: datetime = kwargs.get('create_time', None) or datetime.utcnow()
        self.last_login: Optional[datetime] = kwargs.get('last_login', None)
        self.text: str = text or ''
        self.update_time: datetime = kwargs.get('update_time', None) or datetime.utcnow()
        self.email_verified: bool = kwargs.get('email_verified', False)

    @property
    def domain(self) -> Optional[str]:
        try:
            if '\\' in self.login:
                return self.login.split('\\')[0]
            else:
                return self.email.split('@')[1] if self.email else None
        except (IndexError, AttributeError):
            return None

    @property
    def is_active(self) -> bool:
        return self.status == UserStatus.Active

    @classmethod
    def parse(cls, json: JSON) -> "User":
        return User(
            id=json.get('id', None),  # type: ignore
            name=json['name'],
            login=json.get('login', None) or json.get('email', None),
            password=utils.generate_password_hash(json.get('password', '')),
            email=json.get('email', None),
            status=json.get('status', None),
            roles=json.get('roles', list()),
            attributes=json.get('attributes', dict()),
            text=json.get('text', None),
            email_verified=json.get('email_verified', None)
        )

    def verify_password(self, password: str) -> bool:
        return utils.check_password_hash(self.password, password)

    @property
    def serialize(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'href': absolute_url('/user/' + self.id),
            'name': self.name,
            'login': self.login,
            'email': self.email,
            'domain': self.domain,
            'status': self.status,
            'roles': self.roles,
            'attributes': self.attributes,
            'createTime': self.create_time,
            'lastLogin': self.last_login,
            'text': self.text,
            'updateTime': self.update_time,
            'email_verified': self.email_verified or False
        }

    def __repr__(self) -> str:
        return 'User(id={!r}, name={!r}, login={!r}, status={!r}, roles={!r}, email_verified={!r})'.format(
            self.id, self.name, self.login, self.status, ','.join(self.roles), self.email_verified
        )

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "User":
        return User(
            id=doc.get('id', None) or doc.get('_id'),
            name=doc.get('name', None),
            login=doc.get('login', None) or doc.get('email', None) or 'n/a',
            password=doc.get('password', None),
            email=doc.get('email', None),
            status=doc.get('status', None),
            roles=doc.get('roles', list()),
            attributes=doc.get('attributes', dict()),
            create_time=doc.get('createTime', None),
            last_login=doc.get('lastLogin', None),
            text=doc.get('text', None),
            update_time=doc.get('updateTime', None),
            email_verified=doc.get('email_verified', None)
        )

    @classmethod
    def from_record(cls, rec: Any) -> "User":
        return User(
            id=rec.id,
            name=rec.name,
            login=rec.login or rec.email or 'n/a',
            password=rec.password,
            email=rec.email,
            status=rec.status,
            roles=rec.roles,
            attributes=dict(rec.attributes),
            create_time=rec.create_time,
            last_login=rec.last_login,
            text=rec.text,
            update_time=rec.update_time,
            email_verified=rec.email_verified
        )

    @classmethod
    def from_db(cls, r: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional["User"]:
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        return None

    def create(self) -> "User":
        return User.from_db(db.create_user(self))  # type: ignore

    @staticmethod
    def find_by_id(id: str) -> Optional["User"]:
        return User.from_db(db.get_user(id))

    @staticmethod
    def find_by_username(username: str) -> Optional["User"]:
        """A username may be a login id or an email address."""
        return User.from_db(db.get_user_by_username(username))

    @staticmethod
    def find_by_email(email: str) -> Optional["User"]:
        return User.from_db(db.get_user_by_email(email))

    @staticmethod
    def find_all(query: Optional[Query] = None, page: int = 1, page_size: int = 1000) -> List["User"]:
        return [User.from_db(user) for user in db.get_users(query, page, page_size)]  # type: ignore

    @staticmethod
    def count(query: Optional[Query] = None) -> int:
        return db.get_users_count(query)

    def update_last_login(self) -> Any:
        return db.update_last_login(self.id)

    def update(self, **kwargs: Any) -> "User":
        if kwargs.get('email', None) is not None:
            if '@' not in kwargs['email']:
                raise ValueError(f'Value for "email" not valid: {kwargs["email"]}')
            kwargs['email_verified'] = kwargs.get('email_verified', False)
        if kwargs.get('password') is not None:
            kwargs['password'] = utils.generate_password_hash(kwargs['password'])
        if 'role' in kwargs:
            kwargs['roles'] = [kwargs['role']]
        return User.from_db(db.update_user(self.id, **kwargs))  # type: ignore

    def update_attributes(self, attributes: Dict[str, Any]) -> Any:
        return db.update_user_attributes(self.id, self.attributes, attributes)

    def delete(self) -> Any:
        return db.delete_user(self.id)

    def get_groups(self) -> List[Group]:
        return [Group.from_db(g) for g in db.get_groups_by_user(self.id)]

    @staticmethod
    def check_credentials(username: str, password: str) -> Optional["User"]:
        user = User.find_by_username(username)
        if user and user.verify_password(password):
            return user
        return None

    @staticmethod
    def verify_hash(hash: str, salt: Optional[str] = None) -> Optional["User"]:
        utils.confirm_email_token(hash, salt)
        return User.from_db(db.get_user_by_hash(hash))

    def _set_email_hash(self, hash: str) -> Any:
        return db.set_email_hash(self.id, hash)

    def _clear_email_hash(self) -> Any:
        return db.set_email_hash(self.id, hash=None)

    def send_confirmation(self) -> None:
        token: str = utils.generate_email_token(email=self.email, salt='confirm')  # type: ignore
        self._set_email_hash(token)
        utils.send_confirmation(self, token)

    def set_email_verified(self, verified: bool = True) -> None:
        self.update(email_verified=verified)
        self._clear_email_hash()

    def send_password_reset(self) -> None:
        token: str = utils.generate_email_token(email=self.email, salt='reset')  # type: ignore
        self._set_email_hash(token)
        utils.send_password_reset(self, token)

    def reset_password(self, password: str) -> None:
        self.update(password=password)
        self._clear_email_hash()