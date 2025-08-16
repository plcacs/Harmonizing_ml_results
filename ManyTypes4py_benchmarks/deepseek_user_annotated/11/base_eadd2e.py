from importlib import import_module
from typing import NamedTuple, Optional, Dict, Any, List, Type, Union, Tuple, TypeVar
from urllib.parse import urlparse
from flask import Flask, g
from pkg_resources import iter_entry_points
from werkzeug.local import LocalProxy

# http://stackoverflow.com/questions/8544983/dynamically-mixin-a-base-class-to-an-instance-in-python

T = TypeVar('T')

class Query(NamedTuple):
    where: str
    sort: str
    group: str


class Base:
    pass


def get_backend(app: Flask) -> str:
    db_uri = app.config['DATABASE_URL']
    backend = urlparse(db_uri).scheme

    if backend.startswith('mongodb'):
        backend = 'mongodb'
    if backend == 'postgresql':
        backend = 'postgres'
    return backend


def load_backend(backend: str) -> Any:
    for ep in iter_entry_points('alerta.database.backends'):
        if ep.name == backend:
            module_name = ep.module_name
            break
    else:
        module_name = f'alerta.database.backends.{backend}'

    try:
        return import_module(module_name)
    except Exception:
        raise ImportError(f'Failed to load {backend} database backend')


class Database(Base):

    def __init__(self, app: Optional[Flask] = None) -> None:
        self.app = None
        if app is not None:
            self.init_db(app)

    def init_db(self, app: Flask) -> None:
        backend = get_backend(app)
        cls = load_backend(backend)
        self.__class__ = type('DatabaseImpl', (cls.Backend, Database), {})

        try:
            self.create_engine(app, uri=app.config['DATABASE_URL'], dbname=app.config['DATABASE_NAME'], schema=app.config['DATABASE_SCHEMA'],
                               raise_on_error=app.config['DATABASE_RAISE_ON_ERROR'])
        except Exception as e:
            if app.config['DATABASE_RAISE_ON_ERROR']:
                raise
            app.logger.warning(e)

        app.teardown_appcontext(self.teardown_db)

    def create_engine(self, app: Flask, uri: str, dbname: Optional[str] = None, schema: Optional[str] = None, raise_on_error: bool = True) -> None:
        raise NotImplementedError('Database engine has no create_engine() method')

    def connect(self) -> Any:
        raise NotImplementedError('Database engine has no connect() method')

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def version(self) -> str:
        raise NotImplementedError

    @property
    def is_alive(self) -> bool:
        raise NotImplementedError

    def close(self, db: Any) -> None:
        raise NotImplementedError('Database engine has no close() method')

    def destroy(self) -> None:
        raise NotImplementedError('Database engine has no destroy() method')

    def get_db(self) -> LocalProxy:
        if 'db' not in g:
            g.db = self.connect()
        return g.db

    def teardown_db(self, exc: Optional[BaseException]) -> None:
        db = g.pop('db', None)
        if db is not None:
            self.close(db)

    # ALERTS

    def get_severity(self, alert: Any) -> str:
        raise NotImplementedError

    def get_status(self, alert: Any) -> str:
        raise NotImplementedError

    def is_duplicate(self, alert: Any) -> bool:
        raise NotImplementedError

    def is_correlated(self, alert: Any) -> bool:
        raise NotImplementedError

    def is_flapping(self, alert: Any, window: int = 1800, count: int = 2) -> bool:
        raise NotImplementedError

    def dedup_alert(self, alert: Any, history: List[Any]) -> Any:
        raise NotImplementedError

    def correlate_alert(self, alert: Any, history: List[Any]) -> Any:
        raise NotImplementedError

    def create_alert(self, alert: Any) -> Any:
        raise NotImplementedError

    def set_alert(self, id: str, severity: str, status: str, tags: List[str], attributes: Dict[str, Any], timeout: int, previous_severity: str, update_time: Any, history: Optional[List[Any]] = None) -> Any:
        raise NotImplementedError

    def get_alert(self, id: str, customers: Optional[List[str]] = None) -> Any:
        raise NotImplementedError

    # STATUS, TAGS, ATTRIBUTES

    def set_status(self, id: str, status: str, timeout: int, update_time: Any, history: Optional[List[Any]] = None) -> Any:
        raise NotImplementedError

    def tag_alert(self, id: str, tags: List[str]) -> Any:
        raise NotImplementedError

    def untag_alert(self, id: str, tags: List[str]) -> Any:
        raise NotImplementedError

    def update_tags(self, id: str, tags: List[str]) -> Any:
        raise NotImplementedError

    def update_attributes(self, id: str, old_attrs: Dict[str, Any], new_attrs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def add_history(self, id: str, history: Any) -> Any:
        raise NotImplementedError

    def delete_alert(self, id: str) -> bool:
        raise NotImplementedError

    # BULK

    def tag_alerts(self, query: Optional[Query] = None, tags: Optional[List[str]] = None) -> int:
        raise NotImplementedError

    def untag_alerts(self, query: Optional[Query] = None, tags: Optional[List[str]] = None) -> int:
        raise NotImplementedError

    def update_attributes_by_query(self, query: Optional[Query] = None, attributes: Optional[Dict[str, Any]] = None) -> int:
        raise NotImplementedError

    def delete_alerts(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    # SEARCH & HISTORY

    def get_alerts(self, query: Optional[Query] = None, raw_data: bool = False, history: bool = False, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_alert_history(self, alert: Any, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_history(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    # COUNTS

    def get_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def get_counts(self, query: Optional[Query] = None, group: Optional[str] = None) -> Dict[str, int]:
        raise NotImplementedError

    def get_counts_by_severity(self, query: Optional[Query] = None) -> Dict[str, int]:
        raise NotImplementedError

    def get_counts_by_status(self, query: Optional[Query] = None) -> Dict[str, int]:
        raise NotImplementedError

    def get_topn_count(self, query: Query, group: str = 'event', topn: int = 100) -> List[Tuple[str, int]]:
        raise NotImplementedError

    def get_topn_flapping(self, query: Query, group: str = 'event', topn: int = 100) -> List[Tuple[str, int]]:
        raise NotImplementedError

    def get_topn_standing(self, query: Query, group: str = 'event', topn: int = 100) -> List[Tuple[str, int]]:
        raise NotImplementedError

    # ENVIRONMENTS

    def get_environments(self, query: Optional[Query] = None, topn: int = 1000) -> List[str]:
        raise NotImplementedError

    # SERVICES

    def get_services(self, query: Optional[Query] = None, topn: int = 1000) -> List[str]:
        raise NotImplementedError

    # ALERT GROUPS

    def get_alert_groups(self, query: Optional[Query] = None, topn: int = 1000) -> List[str]:
        raise NotImplementedError

    # ALERT TAGS

    def get_alert_tags(self, query: Optional[Query] = None, topn: int = 1000) -> List[str]:
        raise NotImplementedError

    # BLACKOUTS

    def create_blackout(self, blackout: Any) -> Any:
        raise NotImplementedError

    def get_blackout(self, id: str, customers: Optional[List[str]] = None) -> Any:
        raise NotImplementedError

    def get_blackouts(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_blackouts_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def is_blackout_period(self, alert: Any) -> bool:
        raise NotImplementedError

    def update_blackout(self, id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def delete_blackout(self, id: str) -> bool:
        raise NotImplementedError

    # HEARTBEATS

    def upsert_heartbeat(self, heartbeat: Any) -> Any:
        raise NotImplementedError

    def get_heartbeat(self, id: str, customers: Optional[List[str]] = None) -> Any:
        raise NotImplementedError

    def get_heartbeats(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_heartbeats_by_status(self, status: Optional[str] = None, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_heartbeats_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def delete_heartbeat(self, id: str) -> bool:
        raise NotImplementedError

    # API KEYS

    def create_key(self, key: Any) -> Any:
        raise NotImplementedError

    def get_key(self, key: str, user: Optional[str] = None) -> Any:
        raise NotImplementedError

    def get_keys(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_keys_by_user(self, user: str) -> List[Any]:
        raise NotImplementedError

    def get_keys_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def update_key(self, key: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def update_key_last_used(self, key: str) -> Any:
        raise NotImplementedError

    def delete_key(self, key: str) -> bool:
        raise NotImplementedError

    # USERS

    def create_user(self, user: Any) -> Any:
        raise NotImplementedError

    def get_user(self, id: str) -> Any:
        raise NotImplementedError

    def get_users(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_users_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def get_user_by_username(self, username: str) -> Any:
        raise NotImplementedError

    def get_user_by_email(self, email: str) -> Any:
        raise NotImplementedError

    def get_user_by_hash(self, hash: str) -> Any:
        raise NotImplementedError

    def update_last_login(self, id: str) -> Any:
        raise NotImplementedError

    def update_user(self, id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def update_user_attributes(self, id: str, old_attrs: Dict[str, Any], new_attrs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def delete_user(self, id: str) -> bool:
        raise NotImplementedError

    def set_email_hash(self, id: str, hash: str) -> Any:
        raise NotImplementedError

    # GROUPS

    def create_group(self, group: Any) -> Any:
        raise NotImplementedError

    def get_group(self, id: str) -> Any:
        raise NotImplementedError

    def get_group_users(self, id: str) -> List[Any]:
        raise NotImplementedError

    def get_groups(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_groups_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def update_group(self, id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def add_user_to_group(self, group: str, user: str) -> Any:
        raise NotImplementedError

    def remove_user_from_group(self, group: str, user: str) -> Any:
        raise NotImplementedError

    def delete_group(self, id: str) -> bool:
        raise NotImplementedError

    def get_groups_by_user(self, user: str) -> List[Any]:
        raise NotImplementedError

    # PERMISSIONS

    def create_perm(self, perm: Any) -> Any:
        raise NotImplementedError

    def get_perm(self, id: str) -> Any:
        raise NotImplementedError

    def get_perms(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_perms_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def update_perm(self, id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def delete_perm(self, id: str) -> bool:
        raise NotImplementedError

    def get_scopes_by_match(self, login: str, matches: List[str]) -> List[str]:
        raise NotImplementedError

    # CUSTOMERS

    def create_customer(self, customer: Any) -> Any:
        raise NotImplementedError

    def get_customer(self, id: str) -> Any:
        raise NotImplementedError

    def get_customers(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_customers_count(self, query: Optional[Query] = None) -> int:
        raise NotImplementedError

    def update_customer(self, id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def delete_customer(self, id: str) -> bool:
        raise NotImplementedError

    def get_customers_by_match(self, login: str, matches: List[str]) -> List[str]:
        raise NotImplementedError

    # NOTES

    def create_note(self, note: Any) -> Any:
        raise NotImplementedError

    def get_note(self, id: str) -> Any:
        raise NotImplementedError

    def get_notes(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_alert_notes(self, id: str, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def get_customer_notes(self, id: str, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def update_note(self, id: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def delete_note(self, id: str) -> bool:
        raise NotImplementedError

    # METRICS

    def get_metrics(self, type: Optional[str] = None) -> List[Any]:
        raise NotImplementedError

    def set_gauge(self, gauge: Any) -> Any:
        raise NotImplementedError

    def inc_counter(self, counter: Any) -> Any:
        raise NotImplementedError

    def update_timer(self, timer: Any) -> Any:
        raise NotImplementedError

    # HOUSEKEEPING

    def get_expired(self, expired_threshold: int, info_threshold: int) -> List[Any]:
        raise NotImplementedError

    def get_unshelve(self) -> List[Any]:
        raise NotImplementedError

    def get_unack(self) -> List[Any]:
        raise NotImplementedError


class QueryBuilder(Base):

    def __init__(self, app: Optional[Flask] = None) -> None:
        self.app = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        backend = get_backend(app)
        cls = load_backend(backend)

        self.__class__.alerts = type('AlertsQueryBuilder', (cls.Alerts, self.Alerts, QueryBuilder), {})
        self.__class__.blackouts = type('BlackoutsQueryBuilder', (cls.Blackouts, self.Blackouts, QueryBuilder), {})
        self.__class__.heartbeats = type('HeartbeatsQueryBuilder', (cls.Heartbeats, self.Heartbeats, QueryBuilder), {})
        self.__class__.keys = type('ApiKeysQueryBuilder', (cls.ApiKeys, self.ApiKeys, QueryBuilder), {})
        self.__class__.users = type('UsersQueryBuilder', (cls.Users, self.Users, QueryBuilder), {})
        self.__class__.groups = type('GroupsQueryBuilder', (cls.Groups, self.Groups, QueryBuilder), {})
        self.__class__.perms = type('PermissionsQueryBuilder', (cls.Permissions, self.Permissions, QueryBuilder), {})
        self.__class__.customers = type('CustomersQueryBuilder', (cls.Customers, self.Customers, QueryBuilder), {})

    class Alerts:

        @