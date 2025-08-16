from importlib import import_module
from typing import NamedTuple, Optional, Any, Dict, List, Tuple
from urllib.parse import urlparse
from flask import g, Flask
from pkg_resources import iter_entry_points

class Query(NamedTuple):
    pass

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
        self.app: Optional[Flask] = None
        if app is not None:
            self.init_db(app)

    def init_db(self, app: Flask) -> None:
        backend = get_backend(app)
        cls = load_backend(backend)
        self.__class__ = type('DatabaseImpl', (cls.Backend, Database), {})
        try:
            self.create_engine(app, uri=app.config['DATABASE_URL'], dbname=app.config['DATABASE_NAME'], schema=app.config['DATABASE_SCHEMA'], raise_on_error=app.config['DATABASE_RAISE_ON_ERROR'])
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
    def name(self) -> Any:
        raise NotImplementedError

    @property
    def version(self) -> Any:
        raise NotImplementedError

    @property
    def is_alive(self) -> Any:
        raise NotImplementedError

    def close(self, db: Any) -> None:
        raise NotImplementedError('Database engine has no close() method')

    def destroy(self) -> None:
        raise NotImplementedError('Database engine has no destroy() method')

    def get_db(self) -> Any:
        if 'db' not in g:
            g.db = self.connect()
        return g.db

    def teardown_db(self, exc: Optional[Exception]) -> None:
        db = g.pop('db', None)
        if db is not None:
            self.close(db)

    def get_severity(self, alert: Any) -> Any:
        raise NotImplementedError

    # Rest of the methods follow the same pattern with NotImplementedError return type

class QueryBuilder(Base):

    def __init__(self, app: Optional[Flask] = None) -> None:
        self.app: Optional[Flask] = None
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

        @staticmethod
        def from_params(params: Dict[str, Any], customers: Optional[List[str] = None, query_time: Optional[int] = None) -> Any:
            raise NotImplementedError('AlertsQueryBuilder has no from_params() method for alerts')

    # Rest of the inner classes follow the same pattern with NotImplementedError return type
