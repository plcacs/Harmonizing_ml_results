from importlib import import_module
from typing import NamedTuple
from urllib.parse import urlparse
from flask import g
from pkg_resources import iter_entry_points

class Query(NamedTuple):
    pass

class Base:
    pass

def get_backend(app):
    db_uri = app.config['DATABASE_URL']
    backend = urlparse(db_uri).scheme
    if backend.startswith('mongodb'):
        backend = 'mongodb'
    if backend == 'postgresql':
        backend = 'postgres'
    return backend

def load_backend(backend):
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

    def __init__(self, app=None):
        self.app = None
        if app is not None:
            self.init_db(app)

    def init_db(self, app):
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

    def create_engine(self, app, uri, dbname=None, schema=None, raise_on_error=True):
        raise NotImplementedError('Database engine has no create_engine() method')

    def connect(self):
        raise NotImplementedError('Database engine has no connect() method')

    @property
    def name(self):
        raise NotImplementedError

    @property
    def version(self):
        raise NotImplementedError

    @property
    def is_alive(self):
        raise NotImplementedError

    def close(self, db):
        raise NotImplementedError('Database engine has no close() method')

    def destroy(self):
        raise NotImplementedError('Database engine has no destroy() method')

    def get_db(self):
        if 'db' not in g:
            g.db = self.connect()
        return g.db

    def teardown_db(self, exc):
        db = g.pop('db', None)
        if db is not None:
            self.close(db)

    def get_severity(self, alert):
        raise NotImplementedError

    def get_status(self, alert):
        raise NotImplementedError

    def is_duplicate(self, alert):
        raise NotImplementedError

    def is_correlated(self, alert):
        raise NotImplementedError

    def is_flapping(self, alert, window=1800, count=2):
        raise NotImplementedError

    def dedup_alert(self, alert, history):
        raise NotImplementedError

    def correlate_alert(self, alert, history):
        raise NotImplementedError

    def create_alert(self, alert):
        raise NotImplementedError

    def set_alert(self, id, severity, status, tags, attributes, timeout, previous_severity, update_time, history=None):
        raise NotImplementedError

    def get_alert(self, id, customers=None):
        raise NotImplementedError

    def set_status(self, id, status, timeout, update_time, history=None):
        raise NotImplementedError

    def tag_alert(self, id, tags):
        raise NotImplementedError

    def untag_alert(self, id, tags):
        raise NotImplementedError

    def update_tags(self, id, tags):
        raise NotImplementedError

    def update_attributes(self, id, old_attrs, new_attrs):
        raise NotImplementedError

    def add_history(self, id, history):
        raise NotImplementedError

    def delete_alert(self, id):
        raise NotImplementedError

    def tag_alerts(self, query=None, tags=None):
        raise NotImplementedError

    def untag_alerts(self, query=None, tags=None):
        raise NotImplementedError

    def update_attributes_by_query(self, query=None, attributes=None):
        raise NotImplementedError

    def delete_alerts(self, query=None):
        raise NotImplementedError

    def get_alerts(self, query=None, raw_data=False, history=False, page=None, page_size=None):
        raise NotImplementedError

    def get_alert_history(self, alert, page=None, page_size=None):
        raise NotImplementedError

    def get_history(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_count(self, query=None):
        raise NotImplementedError

    def get_counts(self, query=None, group=None):
        raise NotImplementedError

    def get_counts_by_severity(self, query=None):
        raise NotImplementedError

    def get_counts_by_status(self, query=None):
        raise NotImplementedError

    def get_topn_count(self, query, group='event', topn=100):
        raise NotImplementedError

    def get_topn_flapping(self, query, group='event', topn=100):
        raise NotImplementedError

    def get_topn_standing(self, query, group='event', topn=100):
        raise NotImplementedError

    def get_environments(self, query=None, topn=1000):
        raise NotImplementedError

    def get_services(self, query=None, topn=1000):
        raise NotImplementedError

    def get_alert_groups(self, query=None, topn=1000):
        raise NotImplementedError

    def get_alert_tags(self, query=None, topn=1000):
        raise NotImplementedError

    def create_blackout(self, blackout):
        raise NotImplementedError

    def get_blackout(self, id, customers=None):
        raise NotImplementedError

    def get_blackouts(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_blackouts_count(self, query=None):
        raise NotImplementedError

    def is_blackout_period(self, alert):
        raise NotImplementedError

    def update_blackout(self, id, **kwargs):
        raise NotImplementedError

    def delete_blackout(self, id):
        raise NotImplementedError

    def upsert_heartbeat(self, heartbeat):
        raise NotImplementedError

    def get_heartbeat(self, id, customers=None):
        raise NotImplementedError

    def get_heartbeats(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_heartbeats_by_status(self, status=None, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_heartbeats_count(self, query=None):
        raise NotImplementedError

    def delete_heartbeat(self, id):
        raise NotImplementedError

    def create_key(self, key):
        raise NotImplementedError

    def get_key(self, key, user=None):
        raise NotImplementedError

    def get_keys(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_keys_by_user(self, user):
        raise NotImplementedError

    def get_keys_count(self, query=None):
        raise NotImplementedError

    def update_key(self, key, **kwargs):
        raise NotImplementedError

    def update_key_last_used(self, key):
        raise NotImplementedError

    def delete_key(self, key):
        raise NotImplementedError

    def create_user(self, user):
        raise NotImplementedError

    def get_user(self, id):
        raise NotImplementedError

    def get_users(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_users_count(self, query=None):
        raise NotImplementedError

    def get_user_by_username(self, username):
        raise NotImplementedError

    def get_user_by_email(self, email):
        raise NotImplementedError

    def get_user_by_hash(self, hash):
        raise NotImplementedError

    def update_last_login(self, id):
        raise NotImplementedError

    def update_user(self, id, **kwargs):
        raise NotImplementedError

    def update_user_attributes(self, id, old_attrs, new_attrs):
        raise NotImplementedError

    def delete_user(self, id):
        raise NotImplementedError

    def set_email_hash(self, id, hash):
        raise NotImplementedError

    def create_group(self, group):
        raise NotImplementedError

    def get_group(self, id):
        raise NotImplementedError

    def get_group_users(self, id):
        raise NotImplementedError

    def get_groups(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_groups_count(self, query=None):
        raise NotImplementedError

    def update_group(self, id, **kwargs):
        raise NotImplementedError

    def add_user_to_group(self, group, user):
        raise NotImplementedError

    def remove_user_from_group(self, group, user):
        raise NotImplementedError

    def delete_group(self, id):
        raise NotImplementedError

    def get_groups_by_user(self, user):
        raise NotImplementedError

    def create_perm(self, perm):
        raise NotImplementedError

    def get_perm(self, id):
        raise NotImplementedError

    def get_perms(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_perms_count(self, query=None):
        raise NotImplementedError

    def update_perm(self, id, **kwargs):
        raise NotImplementedError

    def delete_perm(self, id):
        raise NotImplementedError

    def get_scopes_by_match(self, login, matches):
        raise NotImplementedError

    def create_customer(self, customer):
        raise NotImplementedError

    def get_customer(self, id):
        raise NotImplementedError

    def get_customers(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_customers_count(self, query=None):
        raise NotImplementedError

    def update_customer(self, id, **kwargs):
        raise NotImplementedError

    def delete_customer(self, id):
        raise NotImplementedError

    def get_customers_by_match(self, login, matches):
        raise NotImplementedError

    def create_note(self, note):
        raise NotImplementedError

    def get_note(self, id):
        raise NotImplementedError

    def get_notes(self, query=None, page=None, page_size=None):
        raise NotImplementedError

    def get_alert_notes(self, id, page=None, page_size=None):
        raise NotImplementedError

    def get_customer_notes(self, id, page=None, page_size=None):
        raise NotImplementedError

    def update_note(self, id, **kwargs):
        raise NotImplementedError

    def delete_note(self, id):
        raise NotImplementedError

    def get_metrics(self, type=None):
        raise NotImplementedError

    def set_gauge(self, gauge):
        raise NotImplementedError

    def inc_counter(self, counter):
        raise NotImplementedError

    def update_timer(self, timer):
        raise NotImplementedError

    def get_expired(self, expired_threshold, info_threshold):
        raise NotImplementedError

    def get_unshelve(self):
        raise NotImplementedError

    def get_unack(self):
        raise NotImplementedError

class QueryBuilder(Base):

    def __init__(self, app=None):
        self.app = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
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
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('AlertsQueryBuilder has no from_params() method for alerts')

    class Blackouts:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('BlackoutsQueryBuilder has no from_params() method')

    class Heartbeats:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('HeartbeatsQueryBuilder has no from_params() method')

    class ApiKeys:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('ApiKeysQueryBuilder has no from_params() method')

    class Users:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('UsersQueryBuilder has no from_params() method')

    class Groups:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('GroupsQueryBuilder has no from_params() method')

    class Permissions:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('PermissionsQueryBuilder has no from_params() method')

    class Customers:

        @staticmethod
        def from_params(params, customers=None, query_time=None):
            raise NotImplementedError('CustomersQueryBuilder has no from_params() method')