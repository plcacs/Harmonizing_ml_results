from __future__ import annotations
import logging
import textwrap
from typing import Any, Callable, cast, TYPE_CHECKING
import pandas as pd
import sqlalchemy as sqla
from flask_appbuilder import Model
from sqlalchemy import Boolean, Column, create_engine, DateTime, ForeignKey, Integer, MetaData, String, Table as SqlaTable, Text
from sqlalchemy.engine import Connection, Dialect, Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import ColumnElement, expression, Select

from superset.models.helpers import AuditMixinNullable, ImportExportMixin, UUIDMixin
from superset.result_set import SupersetResultSet
from superset.sql_parse import Table
from superset.superset_typing import DbapiDescription, OAuth2ClientConfig, ResultSetColumnType
from superset.utils.backports import StrEnum

if TYPE_CHECKING:
    from superset.databases.ssh_tunnel.models import SSHTunnel
    from superset.models.sql_lab import Query

class KeyValue(Model):
    __tablename__ = 'keyvalue'
    id: int = Column(Integer, primary_key=True)
    value: str = Column(utils.MediumText(), nullable=False)

class CssTemplate(AuditMixinNullable, UUIDMixin, Model):
    __tablename__ = 'css_templates'
    id: int = Column(Integer, primary_key=True)
    template_name: str = Column(String(250))
    css: str = Column(utils.MediumText(), default='')

class ConfigurationMethod(StrEnum):
    SQLALCHEMY_FORM = 'sqlalchemy_form'
    DYNAMIC_FORM = 'dynamic_form'

class Database(Model, AuditMixinNullable, ImportExportMixin):
    __tablename__ = 'dbs'
    type: str = 'table'
    __table_args__ = (UniqueConstraint('database_name'),)
    id: int = Column(Integer, primary_key=True)
    verbose_name: str = Column(String(250), unique=True)
    database_name: str = Column(String(250), unique=True, nullable=False)
    sqlalchemy_uri: str = Column(String(1024), nullable=False)
    password: str = Column(encrypted_field_factory.create(String(1024)))
    cache_timeout: int = Column(Integer)
    select_as_create_table_as: bool = Column(Boolean, default=False)
    expose_in_sqllab: bool = Column(Boolean, default=True)
    configuration_method: str = Column(String(255), server_default=ConfigurationMethod.SQLALCHEMY_FORM.value)
    allow_run_async: bool = Column(Boolean, default=False)
    allow_file_upload: bool = Column(Boolean, default=False)
    allow_ctas: bool = Column(Boolean, default=False)
    allow_cvas: bool = Column(Boolean, default=False)
    allow_dml: bool = Column(Boolean, default=False)
    force_ctas_schema: str = Column(String(250))
    extra: str = Column(Text, default=textwrap.dedent('    {\n        "metadata_params": {},\n        "engine_params": {},\n        "metadata_cache_timeout": {},\n        "schemas_allowed_for_file_upload": []\n    }\n    '))
    encrypted_extra: str = Column(encrypted_field_factory.create(Text), nullable=True)
    impersonate_user: bool = Column(Boolean, default=False)
    server_cert: str = Column(encrypted_field_factory.create(Text), nullable=True)
    is_managed_externally: bool = Column(Boolean, nullable=False, default=False)
    external_url: str = Column(Text, nullable=True)
    export_fields: list[str] = ['database_name', 'sqlalchemy_uri', 'cache_timeout', 'expose_in_sqllab', 'allow_run_async', 'allow_ctas', 'allow_cvas', 'allow_dml', 'allow_file_upload', 'extra', 'impersonate_user']
    extra_import_fields: list[str] = ['password', 'is_managed_externally', 'external_url', 'encrypted_extra', 'impersonate_user']
    export_children: list[str] = ['tables']

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self.verbose_name if self.verbose_name else self.database_name

    @property
    def allows_subquery(self) -> bool:
        return self.db_engine_spec.allows_subqueries

    @property
    def function_names(self) -> list[str]:
        try:
            return self.db_engine_spec.get_function_names(self)
        except Exception as ex:
            logger.error('Failed to fetch database function names with error: %s', str(ex), exc_info=True)
        return []

    @property
    def allows_cost_estimate(self) -> bool:
        extra = self.get_extra() or {}
        cost_estimate_enabled = extra.get('cost_estimate_enabled')
        return self.db_engine_spec.get_allow_cost_estimate(extra) and cost_estimate_enabled

    @property
    def allows_virtual_table_explore(self) -> bool:
        extra = self.get_extra()
        return bool(extra.get('allows_virtual_table_explore', True))

    @property
    def explore_database_id(self) -> int:
        return self.get_extra().get('explore_database_id', self.id)

    @property
    def disable_data_preview(self) -> bool:
        return self.get_extra().get('disable_data_preview', False) is True

    @property
    def disable_drill_to_detail(self) -> bool:
        return self.get_extra().get('disable_drill_to_detail', False) is True

    @property
    def allow_multi_catalog(self) -> bool:
        return self.get_extra().get('allow_multi_catalog', False)

    @property
    def schema_options(self) -> dict:
        return self.get_extra().get('schema_options', {})

    @property
    def data(self) -> dict:
        return {'id': self.id, 'name': self.database_name, 'backend': self.backend, 'configuration_method': self.configuration_method, 'allows_subquery': self.allows_subquery, 'allows_cost_estimate': self.allows_cost_estimate, 'allows_virtual_table_explore': self.allows_virtual_table_explore, 'explore_database_id': self.explore_database_id, 'schema_options': self.schema_options, 'parameters': self.parameters, 'disable_data_preview': self.disable_data_preview, 'disable_drill_to_detail': self.disable_drill_to_detail, 'allow_multi_catalog': self.allow_multi_catalog, 'parameters_schema': self.parameters_schema, 'engine_information': self.engine_information}

    @property
    def unique_name(self) -> str:
        return self.database_name

    @property
    def url_object(self):
        return make_url_safe(self.sqlalchemy_uri_decrypted)

    @property
    def backend(self) -> str:
        return self.url_object.get_backend_name()

    @property
    def driver(self) -> str:
        return self.url_object.get_driver_name()

    @property
    def masked_encrypted_extra(self) -> str:
        return self.db_engine_spec.mask_encrypted_extra(self.encrypted_extra)

    @property
    def parameters(self) -> dict:
        masked_uri = make_url_safe(self.sqlalchemy_uri)
        encrypted_config = {}
        if (masked_encrypted_extra := self.masked_encrypted_extra) is not None:
            with suppress(TypeError, json.JSONDecodeError):
                encrypted_config = json.loads(masked_encrypted_extra)
        try:
            parameters = self.db_engine_spec.get_parameters_from_uri(masked_uri, encrypted_extra=encrypted_config)
        except Exception:
            parameters = {}
        return parameters

    @property
    def parameters_schema(self) -> dict:
        try:
            parameters_schema = self.db_engine_spec.parameters_json_schema()
        except Exception:
            parameters_schema = {}
        return parameters_schema

    @property
    def metadata_cache_timeout(self) -> dict:
        return self.get_extra().get('metadata_cache_timeout', {})

    @property
    def catalog_cache_enabled(self) -> bool:
        return 'catalog_cache_timeout' in self.metadata_cache_timeout

    @property
    def catalog_cache_timeout(self) -> int:
        return self.metadata_cache_timeout.get('catalog_cache_timeout')

    @property
    def schema_cache_enabled(self) -> bool:
        return 'schema_cache_timeout' in self.metadata_cache_timeout

    @property
    def schema_cache_timeout(self) -> int:
        return self.metadata_cache_timeout.get('schema_cache_timeout')

    @property
    def table_cache_enabled(self) -> bool:
        return 'table_cache_timeout' in self.metadata_cache_timeout

    @property
    def table_cache_timeout(self) -> int:
        return self.metadata_cache_timeout.get('table_cache_timeout')

    @property
    def default_schemas(self) -> list[str]:
        return self.get_extra().get('default_schemas', [])

    @property
    def connect_args(self) -> dict:
        return self.get_extra().get('engine_params', {}).get('connect_args', {})

    @property
    def engine_information(self) -> dict:
        try:
            engine_information = self.db_engine_spec.get_public_information()
        except Exception:
            engine_information = {}
        return engine_information

    @classmethod
    def get_password_masked_url_from_uri(cls, uri: str) -> URL:
        sqlalchemy_url = make_url_safe(uri)
        return cls.get_password_masked_url(sqlalchemy_url)

    @classmethod
    def get_password_masked_url(cls, masked_url: URL) -> URL:
        url_copy = deepcopy(masked_url)
        if url_copy.password is not None:
            url_copy = url_copy.set(password=PASSWORD_MASK)
        return url_copy

    def set_sqlalchemy_uri(self, uri: str):
        conn = make_url_safe(uri.strip())
        if conn.password != PASSWORD_MASK and (not custom_password_store):
            self.password = conn.password
        conn = conn.set(password=PASSWORD_MASK if conn.password else None)
        self.sqlalchemy_uri = str(conn)

    def get_effective_user(self, object_url: URL) -> str:
        return username if (username := get_username()) else object_url.username if self.impersonate_user else None

    @contextmanager
    def get_sqla_engine(self, catalog=None, schema=None, nullpool=True, source=None, override_ssh_tunnel=None):
        from superset.daos.database import DatabaseDAO
        sqlalchemy_uri = self.sqlalchemy_uri_decrypted
        ssh_tunnel = override_ssh_tunnel or DatabaseDAO.get_ssh_tunnel(self.id)
        ssh_context_manager = ssh_manager_factory.instance.create_tunnel(ssh_tunnel=ssh_tunnel, sqlalchemy_database_uri=sqlalchemy_uri) if ssh_tunnel else nullcontext()
        with ssh_context_manager as ssh_context:
            if ssh_context:
                logger.info('[SSH] Successfully created tunnel w/ %s tunnel_timeout + %s ssh_timeout at %s', sshtunnel.TUNNEL_TIMEOUT, sshtunnel.SSH_TIMEOUT, ssh_context.local_bind_address)
                sqlalchemy_uri = ssh_manager_factory.instance.build_sqla_url(sqlalchemy_uri, ssh_context)
            engine_context_manager = config['ENGINE_CONTEXT_MANAGER']
            with engine_context_manager(self, catalog, schema):
                with check_for_oauth2(self):
                    yield self._get_sqla_engine(catalog=catalog, schema=schema, nullpool=nullpool, source=source, sqlalchemy_uri=sqlalchemy_uri)

    def _get_sqla_engine(self, catalog=None, schema=None, nullpool=True, source=None, sqlalchemy_uri=None) -> Engine:
        sqlalchemy_url = make_url_safe(sqlalchemy_uri if sqlalchemy_uri else self.sqlalchemy_uri_decrypted)
        self.db_engine_spec.validate_database_uri(sqlalchemy_url)
        extra = self.get_extra()
        params = extra.get('engine_params', {})
        if nullpool:
            params['poolclass'] = NullPool
        connect_args = params.get('connect_args', {})
        sqlalchemy_url, connect_args = self.db_engine_spec.adjust_engine_params(uri=sqlalchemy_url, connect_args=connect_args, catalog=catalog, schema=schema)
        effective_username = self.get_effective_user(sqlalchemy_url)
        if effective_username and is_feature_enabled('IMPERSONATE_WITH_EMAIL_PREFIX'):
            user = security_manager.find_user(username=effective_username)
            if user and user.email:
                effective_username = user.email.split('@')[0]
        oauth2_config = self.get_oauth2_config()
        access_token = get_oauth2_access_token(oauth2_config, self.id, g.user.id, self.db_engine_spec) if oauth2_config and hasattr(g, 'user') and hasattr(g.user, 'id') else None
        sqlalchemy_url = self.db_engine_spec.get_url_for_impersonation(sqlalchemy_url, self.impersonate_user, effective_username, access_token)
        masked_url = self.get_password_masked_url(sqlalchemy_url)
        logger.debug('Database._get_sqla_engine(). Masked URL: %s', str(masked_url))
        if self.impersonate_user:
            args = [connect_args, str(sqlalchemy_url), effective_username, access_token]
            args = self.add_database_to_signature(self.db_engine_spec.update_impersonation_config, args)
            self.db_engine_spec.update_impersonation_config(*args)
        if connect_args:
            params['connect_args'] = connect_args
        self.update_params_from_encrypted_extra(params)
        if DB_CONNECTION_MUTATOR:
            if not source and request and request.referrer:
                if '/superset/dashboard/' in request.referrer:
                    source = utils.QuerySource.DASHBOARD
                elif '/explore/' in request.referrer:
                    source = utils.QuerySource.CHART
                elif '/sqllab/' in request.referrer:
                    source = utils.QuerySource.SQL_LAB
            sqlalchemy_url, params = DB_CONNECTION_MUTATOR(sqlalchemy_url, params, effective_username, security_manager, source)
        try:
            return create_engine(sqlalchemy_url, **params)
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    def add_database_to_signature(self, func, args) -> list:
        sig = signature(func)
        if 'database' in (params := sig.parameters.keys()):
            args.insert(list(params).index('database'), self)
        return args

    @contextmanager
    def get_raw_connection(self, catalog=None, schema=None, nullpool=True, source=None):
        with self.get_sqla_engine(catalog=catalog, schema=schema, nullpool=nullpool, source=source) as engine:
            with check_for_oauth2(self):
                with closing(engine.raw_connection()) as conn:
                    for prequery in self.db_engine_spec.get_prequeries(database=self, catalog=catalog, schema=schema):
                        cursor = conn.cursor()
                        cursor.execute(prequery)
                    yield conn

    def get_default_catalog(self) -> str:
        return self.db_engine_spec.get_default_catalog(self)

    def get_default_schema(self, catalog: str) -> str:
        return self.db_engine_spec.get_default_schema(self, catalog)

    def get_default_schema_for_query(self, query: str) -> str:
        return self.db_engine_spec.get_default_schema_for_query(self, query)

    @staticmethod
    def post_process_df(df: pd.DataFrame) -> pd.DataFrame:

        def column_needs_conversion(df_series: pd.Series) -> bool:
            return not df_series.empty and isinstance(df_series, pd.Series) and isinstance(df_series[0], (list, dict))
        for col, coltype in df.dtypes.to_dict().items():
            if coltype == numpy.object_ and column_needs_conversion(df[col]):
                df[col] = df[col].apply(json.json_dumps_w_dates)
        return df

    @property
    def quote_identifier(self) -> Callable:
        return self.get_dialect().identifier_preparer.quote

    def get_reserved_words(self) -> Any:
        return self.get_dialect().preparer.reserved_words

    def mutate_sql_based_on_config(self, sql_: str, limit: int = 1000, force: bool = False) -> str:
        if self.db_engine_spec.allow_limit_clause:
            return self.db_engine_spec.apply_limit_to_sql(sql_, limit, self, force=force)
        return self.db_engine_spec.apply_top_to_sql(sql_, limit)

    def safe_sqlalchemy_uri(self) -> str:
        return self.sqlalchemy_uri

    @cache_util.memoized_func(key='db:{self.id}:catalog:{catalog}:schema:{schema}:table_list', cache=cache_manager.cache)
    def get_all_table_names_in_schema(self, catalog: str, schema: str) -> set:
        try:
            with self.get_inspector(catalog=catalog, schema=schema) as inspector:
                return {(table, schema, catalog) for table in self.db_engine_spec.get_table_names(database=self, inspector=inspector, schema=schema)}
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @cache_util.memoized_func(key='db:{self.id}:catalog:{catalog}:schema:{schema}:view_list', cache=cache_manager.cache)
    def get_all_view_names_in_schema(self, catalog: str, schema: str) -> set:
        try:
            with self.get_inspector(catalog=catalog, schema=schema) as inspector:
                return {(view, schema, catalog) for view in self.db_engine_spec.get_view_names(database=self, inspector=inspector, schema=schema)}
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @contextmanager
    def get_inspector(self, catalog: str = None, schema: str = None, ssh_tunnel: SSHTunnel = None) -> Inspector:
        with self.get_sqla_engine(catalog=catalog, schema=schema, override_ssh_tunnel=ssh_tunnel) as engine:
            yield sqla.inspect(engine)

    @cache_util.memoized_func(key='db:{self.id}:catalog:{catalog}:schema_list', cache=cache_manager.cache)
    def get_all_schema_names(self, *, catalog: str = None, ssh_tunnel: SSHTunnel = None) -> list[str]:
        try:
            with self.get_inspector(catalog=catalog, ssh_tunnel=ssh_tunnel) as inspector:
                return self.db_engine_spec.get_schema_names(inspector)
        except Exception as ex:
            if self.is_oauth2_enabled() and self.db_engine_spec.needs_oauth2(ex):
                self.start_oauth2_dance()
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @cache_util.memoized_func(key='db:{self.id}:catalog_list', cache=cache_manager.cache)
    def get_all_catalog_names(self, *, ssh_tunnel: SSHTunnel = None) -> list[str]:
        try:
            with self.get_inspector(ssh_tunnel=ssh_tunnel) as inspector:
                return self.db_engine_spec.get_catalog_names(self, inspector)
        except Exception as ex:
            if self.is_oauth2_enabled() and self.db_engine_spec.needs_oauth2(ex):
                self.start_oauth2_dance()
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @property
    def db_engine_spec(self) -> MetricType:
        url = make_url_safe(self.sqlalchemy_uri_decrypted)
        return self.get_db_engine_spec(url)

    @classmethod
    @lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
    def get_db_engine_spec(cls, url: URL) -> Engine:
        backend = url.get_backend_name()
        try:
            driver = url.get_driver_name()
        except NoSuchModuleError:
            driver