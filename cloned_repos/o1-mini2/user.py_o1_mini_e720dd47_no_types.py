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

    def __init__(self, name, login, password, email, roles, text, **kwargs: Any
        ):
        if not login:
            raise ValueError('Missing mandatory value for "login"')
        self.id: str = kwargs.get('id') or str(uuid4())
        self.name: str = name or ''
        self.login: str = login
        self.password: str = password
        self.email: str = email
        self.status: UserStatus = kwargs.get('status', None
            ) or UserStatus.Active
        self.roles: List[str] = current_app.config['ADMIN_ROLES'
            ] if self.email and self.email in current_app.config['ADMIN_USERS'
            ] else roles
        self.attributes: Dict[str, Any] = kwargs.get('attributes', None
            ) or dict()
        self.create_time: datetime = kwargs.get('create_time', None
            ) or datetime.utcnow()
        self.last_login: Optional[datetime] = kwargs.get('last_login', None)
        self.text: str = text or ''
        self.update_time: datetime = kwargs.get('update_time', None
            ) or datetime.utcnow()
        self.email_verified: bool = kwargs.get('email_verified', False)

    @property
    def domain(self):
        try:
            if '\\' in self.login:
                return self.login.split('\\')[0]
            else:
                return self.email.split('@')[1]
        except (IndexError, AttributeError):
            return None

    @property
    def is_active(self):
        return self.status == UserStatus.Active

    @classmethod
    def parse(cls, json):
        return User(name=json['name'], login=json.get('login', None) or
            json.get('email', None), password=utils.generate_password_hash(
            json.get('password', '')), email=json.get('email', None), roles
            =json.get('roles', list()), text=json.get('text', None) or '',
            status=json.get('status', None), attributes=json.get(
            'attributes', dict()), email_verified=json.get('email_verified',
            None), **{'id': json.get('id', None)} if 'id' in json else {})

    def verify_password(self, password):
        return utils.check_password_hash(self.password, password)

    @property
    def serialize(self):
        return {'id': self.id, 'href': absolute_url('/user/' + self.id),
            'name': self.name, 'login': self.login, 'email': self.email,
            'domain': self.domain, 'status': self.status, 'roles': self.
            roles, 'attributes': self.attributes, 'createTime': self.
            create_time, 'lastLogin': self.last_login, 'text': self.text,
            'updateTime': self.update_time, 'email_verified': self.
            email_verified or False}

    def __repr__(self):
        return (
            'User(id={!r}, name={!r}, login={!r}, status={!r}, roles={!r}, email_verified={!r})'
            .format(self.id, self.name, self.login, self.status, ','.join(
            self.roles), self.email_verified))

    @classmethod
    def from_document(cls, doc):
        return User(name=doc.get('name', ''), login=doc.get('login', None) or
            doc.get('email', None) or 'n/a', password=doc.get('password',
            ''), email=doc.get('email', ''), roles=doc.get('roles', list()),
            text=doc.get('text', '') or '', status=doc.get('status', None),
            attributes=doc.get('attributes', dict()), create_time=doc.get(
            'createTime', None), last_login=doc.get('lastLogin', None),
            update_time=doc.get('updateTime', None), email_verified=doc.get
            ('email_verified', False), **{'id': doc.get('id', None) or doc.
            get('_id')} if 'id' in doc or '_id' in doc else {})

    @classmethod
    def from_record(cls, rec):
        return User(name=rec.name, login=rec.login or rec.email or 'n/a',
            password=rec.password, email=rec.email, roles=rec.roles, text=
            rec.text or '', status=rec.status, attributes=dict(rec.
            attributes), create_time=rec.create_time, last_login=rec.
            last_login, update_time=rec.update_time, email_verified=rec.
            email_verified, **{'id': rec.id} if hasattr(rec, 'id') else {})

    @classmethod
    def from_db(cls, r):
        if isinstance(r, dict):
            return cls.from_document(r)
        elif isinstance(r, tuple):
            return cls.from_record(r)
        else:
            raise TypeError('Unsupported type for from_db')

    def create(self):
        return User.from_db(db.create_user(self))

    @staticmethod
    def find_by_id(id):
        return User.from_db(db.get_user(id))

    @staticmethod
    def find_by_username(username):
        """A username may be a login id or an email address."""
        return User.from_db(db.get_user_by_username(username))

    @staticmethod
    def find_by_email(email):
        return User.from_db(db.get_user_by_email(email))

    @staticmethod
    def find_all(query=None, page=1, page_size=1000):
        return [User.from_db(user) for user in db.get_users(query, page,
            page_size)]

    @staticmethod
    def count(query=None):
        return db.get_users_count(query)

    def update_last_login(self):
        return db.update_last_login(self.id)

    def update(self, **kwargs: Any):
        if kwargs.get('email') is not None:
            if '@' not in kwargs['email']:
                raise ValueError(
                    f'Value for "email" not valid: {kwargs[\'email\']}')
            kwargs['email_verified'] = kwargs.get('email_verified', False)
        if 'password' in kwargs and kwargs['password'] is not None:
            kwargs['password'] = utils.generate_password_hash(kwargs[
                'password'])
        if 'role' in kwargs:
            kwargs['roles'] = [kwargs['role']]
        return User.from_db(db.update_user(self.id, **kwargs))

    def update_attributes(self, attributes):
        return db.update_user_attributes(self.id, self.attributes, attributes)

    def delete(self):
        return db.delete_user(self.id)

    def get_groups(self):
        return [Group.from_db(g) for g in db.get_groups_by_user(self.id)]

    @staticmethod
    def check_credentials(username, password):
        user = User.find_by_username(username)
        if user and user.verify_password(password):
            return user
        return None

    @staticmethod
    def verify_hash(hash, salt=None):
        utils.confirm_email_token(hash, salt)
        return User.from_db(db.get_user_by_hash(hash))

    def _set_email_hash(self, hash):
        return db.set_email_hash(self.id, hash)

    def _clear_email_hash(self):
        return db.set_email_hash(self.id, hash=None)

    def send_confirmation(self):
        token: str = utils.generate_email_token(email=self.email, salt=
            'confirm')
        self._set_email_hash(token)
        utils.send_confirmation(self, token)

    def set_email_verified(self, verified=True):
        self.update(email_verified=verified)
        self._clear_email_hash()

    def send_password_reset(self):
        token: str = utils.generate_email_token(email=self.email, salt='reset')
        self._set_email_hash(token)
        utils.send_password_reset(self, token)

    def reset_password(self, password):
        self.update(password=password)
        self._clear_email_hash()
