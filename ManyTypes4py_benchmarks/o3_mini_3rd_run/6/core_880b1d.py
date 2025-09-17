from __future__ import annotations
import builtins
import logging
import textwrap
from ast import literal_eval
from contextlib import closing, contextmanager, nullcontext, suppress
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from inspect import signature, Signature
from typing import Any, Callable, cast, Generator, Optional, Set, Tuple, List, Dict, Union, TYPE_CHECKING

import numpy
import pandas as pd
import sqlalchemy as sqla
import sshtunnel
from flask import g, request
from flask_appbuilder import Model
from sqlalchemy import Boolean, Column, create_engine, DateTime, ForeignKey, Integer, MetaData, String, Table as SqlaTable, Text
from sqlalchemy.engine import Connection, Dialect, Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import ColumnElement, expression, Select

from superset import app, db, db_engine_specs, is_feature_enabled
from superset.commands.database.exceptions import DatabaseInvalidError
from superset.constants import LRU_CACHE_MAX_SIZE, PASSWORD_MASK
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.base import MetricType, TimeGrain
from superset.extensions import cache_manager, encrypted_field_factory, event_logger, security_manager, ssh_manager_factory
from superset.models.helpers import AuditMixinNullable, ImportExportMixin, UUIDMixin
from superset.result_set import SupersetResultSet
from superset.sql.parse import SQLScript
from superset.sql_parse import Table
from superset.superset_typing import DbapiDescription, OAuth2ClientConfig, ResultSetColumnType
from superset.utils import cache as cache_util, core as utils, json
from superset.utils.backports import StrEnum
from superset.utils.core import get_username
from superset.utils.oauth2 import check_for_oauth2, get_oauth2_access_token, OAuth2ClientConfigSchema

if TYPE_CHECKING:
    from superset.databases.ssh_tunnel.models import SSHTunnel
    from superset.models.sql_lab import Query

config = app.config
custom_password_store: Any = config['SQLALCHEMY_CUSTOM_PASSWORD_STORE']
stats_logger: Any = config['STATS_LOGGER']
log_query: Any = config['QUERY_LOGGER']
metadata = Model.metadata
logger = logging.getLogger(__name__)
DB_CONNECTION_MUTATOR: Any = config['DB_CONNECTION_MUTATOR']


class KeyValue(Model):
    """Used for any type of key-value store"""
    __tablename__ = 'keyvalue'
    id = Column(Integer, primary_key=True)
    value = Column(utils.MediumText(), nullable=False)


class CssTemplate(AuditMixinNullable, UUIDMixin, Model):
    """CSS templates for dashboards"""
    __tablename__ = 'css_templates'
    id = Column(Integer, primary_key=True)
    template_name = Column(String(250))
    css = Column(utils.MediumText(), default='')


class ConfigurationMethod(StrEnum):
    SQLALCHEMY_FORM = 'sqlalchemy_form'
    DYNAMIC_FORM = 'dynamic_form'


class Database(Model, AuditMixinNullable, ImportExportMixin):
    """An ORM object that stores Database related information"""
    __tablename__ = 'dbs'
    type = 'table'
    __table_args__ = (UniqueConstraint('database_name'),)
    id = Column(Integer, primary_key=True)
    verbose_name = Column(String(250), unique=True)
    database_name = Column(String(250), unique=True, nullable=False)
    sqlalchemy_uri = Column(String(1024), nullable=False)
    password = Column(encrypted_field_factory.create(String(1024)))
    cache_timeout = Column(Integer)
    select_as_create_table_as = Column(Boolean, default=False)
    expose_in_sqllab = Column(Boolean, default=True)
    configuration_method = Column(String(255), server_default=ConfigurationMethod.SQLALCHEMY_FORM.value)
    allow_run_async = Column(Boolean, default=False)
    allow_file_upload = Column(Boolean, default=False)
    allow_ctas = Column(Boolean, default=False)
    allow_cvas = Column(Boolean, default=False)
    allow_dml = Column(Boolean, default=False)
    force_ctas_schema = Column(String(250))
    extra = Column(Text, default=textwrap.dedent('    {\n        "metadata_params": {},\n        "engine_params": {},\n        "metadata_cache_timeout": {},\n        "schemas_allowed_for_file_upload": []\n    }\n    '))
    encrypted_extra = Column(encrypted_field_factory.create(Text), nullable=True)
    impersonate_user = Column(Boolean, default=False)
    server_cert = Column(encrypted_field_factory.create(Text), nullable=True)
    is_managed_externally = Column(Boolean, nullable=False, default=False)
    external_url = Column(Text, nullable=True)
    export_fields: List[str] = ['database_name', 'sqlalchemy_uri', 'cache_timeout', 'expose_in_sqllab', 'allow_run_async', 'allow_ctas', 'allow_cvas', 'allow_dml', 'allow_file_upload', 'extra', 'impersonate_user']
    extra_import_fields: List[str] = ['password', 'is_managed_externally', 'external_url', 'encrypted_extra', 'impersonate_user']
    export_children: List[str] = ['tables']

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self.verbose_name if self.verbose_name else self.database_name

    @property
    def allows_subquery(self) -> Any:
        return self.db_engine_spec.allows_subqueries

    @property
    def function_names(self) -> List[str]:
        try:
            return self.db_engine_spec.get_function_names(self)
        except Exception as ex:
            logger.error('Failed to fetch database function names with error: %s', str(ex), exc_info=True)
        return []

    @property
    def allows_cost_estimate(self) -> bool:
        extra: Dict[str, Any] = self.get_extra() or {}
        cost_estimate_enabled = extra.get('cost_estimate_enabled')
        return self.db_engine_spec.get_allow_cost_estimate(extra) and cost_estimate_enabled

    @property
    def allows_virtual_table_explore(self) -> bool:
        extra = self.get_extra()
        return bool(extra.get('allows_virtual_table_explore', True))

    @property
    def explore_database_id(self) -> Any:
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
    def schema_options(self) -> Dict[str, Any]:
        """Additional schema display config for engines with complex schemas"""
        return self.get_extra().get('schema_options', {})

    @property
    def data(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.database_name,
            'backend': self.backend,
            'configuration_method': self.configuration_method,
            'allows_subquery': self.allows_subquery,
            'allows_cost_estimate': self.allows_cost_estimate,
            'allows_virtual_table_explore': self.allows_virtual_table_explore,
            'explore_database_id': self.explore_database_id,
            'schema_options': self.schema_options,
            'parameters': self.parameters,
            'disable_data_preview': self.disable_data_preview,
            'disable_drill_to_detail': self.disable_drill_to_detail,
            'allow_multi_catalog': self.allow_multi_catalog,
            'parameters_schema': self.parameters_schema,
            'engine_information': self.engine_information,
        }

    @property
    def unique_name(self) -> str:
        return self.database_name

    @property
    def url_object(self) -> URL:
        return make_url_safe(self.sqlalchemy_uri_decrypted)

    @property
    def backend(self) -> str:
        return self.url_object.get_backend_name()

    @property
    def driver(self) -> str:
        return self.url_object.get_driver_name()

    @property
    def masked_encrypted_extra(self) -> Optional[str]:
        return self.db_engine_spec.mask_encrypted_extra(self.encrypted_extra)

    @property
    def parameters(self) -> Dict[str, Any]:
        masked_uri: URL = make_url_safe(self.sqlalchemy_uri)
        encrypted_config: Dict[str, Any] = {}
        if (masked_encrypted_extra := self.masked_encrypted_extra) is not None:
            with suppress(TypeError, json.JSONDecodeError):
                encrypted_config = json.loads(masked_encrypted_extra)
        try:
            parameters = self.db_engine_spec.get_parameters_from_uri(masked_uri, encrypted_extra=encrypted_config)
        except Exception:
            parameters = {}
        return parameters

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        try:
            parameters_schema = self.db_engine_spec.parameters_json_schema()
        except Exception:
            parameters_schema = {}
        return parameters_schema

    @property
    def metadata_cache_timeout(self) -> Any:
        return self.get_extra().get('metadata_cache_timeout', {})

    @property
    def catalog_cache_enabled(self) -> bool:
        return 'catalog_cache_timeout' in self.metadata_cache_timeout

    @property
    def catalog_cache_timeout(self) -> Any:
        return self.metadata_cache_timeout.get('catalog_cache_timeout')

    @property
    def schema_cache_enabled(self) -> bool:
        return 'schema_cache_timeout' in self.metadata_cache_timeout

    @property
    def schema_cache_timeout(self) -> Any:
        return self.metadata_cache_timeout.get('schema_cache_timeout')

    @property
    def table_cache_enabled(self) -> bool:
        return 'table_cache_timeout' in self.metadata_cache_timeout

    @property
    def table_cache_timeout(self) -> Any:
        return self.metadata_cache_timeout.get('table_cache_timeout')

    @property
    def default_schemas(self) -> List[str]:
        return self.get_extra().get('default_schemas', [])

    @property
    def connect_args(self) -> Dict[str, Any]:
        return self.get_extra().get('engine_params', {}).get('connect_args', {})

    @property
    def engine_information(self) -> Dict[str, Any]:
        try:
            engine_information = self.db_engine_spec.get_public_information()
        except Exception:
            engine_information = {}
        return engine_information

    @classmethod
    def get_password_masked_url_from_uri(cls, uri: str) -> URL:
        sqlalchemy_url: URL = make_url_safe(uri)
        return cls.get_password_masked_url(sqlalchemy_url)

    @classmethod
    def get_password_masked_url(cls, masked_url: URL) -> URL:
        url_copy: URL = deepcopy(masked_url)
        if url_copy.password is not None:
            url_copy = url_copy.set(password=PASSWORD_MASK)
        return url_copy

    def set_sqlalchemy_uri(self, uri: str) -> None:
        conn: URL = make_url_safe(uri.strip())
        if conn.password != PASSWORD_MASK and (not custom_password_store):
            self.password = conn.password
        conn = conn.set(password=PASSWORD_MASK if conn.password else None)
        self.sqlalchemy_uri = str(conn)

    def get_effective_user(self, object_url: URL) -> Optional[str]:
        """
        Get the effective user, especially during impersonation.

        :param object_url: SQL Alchemy URL object
        :return: The effective username
        """
        username: Optional[str] = get_username()
        return username if username else object_url.username if self.impersonate_user else None

    @contextmanager
    def get_sqla_engine(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        nullpool: bool = True,
        source: Optional[str] = None,
        override_ssh_tunnel: Optional[SSHTunnel] = None,
    ) -> Generator[Engine, None, None]:
        """
        Context manager for a SQLAlchemy engine.
        """
        from superset.daos.database import DatabaseDAO
        sqlalchemy_uri: str = self.sqlalchemy_uri_decrypted
        ssh_tunnel = override_ssh_tunnel or DatabaseDAO.get_ssh_tunnel(self.id)
        ssh_context_manager = (
            ssh_manager_factory.instance.create_tunnel(
                ssh_tunnel=ssh_tunnel, sqlalchemy_database_uri=sqlalchemy_uri
            )
            if ssh_tunnel
            else nullcontext()
        )
        with ssh_context_manager as ssh_context:
            if ssh_context:
                logger.info(
                    '[SSH] Successfully created tunnel w/ %s tunnel_timeout + %s ssh_timeout at %s',
                    sshtunnel.TUNNEL_TIMEOUT,
                    sshtunnel.SSH_TIMEOUT,
                    ssh_context.local_bind_address,
                )
                sqlalchemy_uri = ssh_manager_factory.instance.build_sqla_url(sqlalchemy_uri, ssh_context)
            engine_context_manager: Callable[..., Any] = config['ENGINE_CONTEXT_MANAGER']
            with engine_context_manager(self, catalog, schema):
                with check_for_oauth2(self):
                    yield self._get_sqla_engine(
                        catalog=catalog, schema=schema, nullpool=nullpool, source=source, sqlalchemy_uri=sqlalchemy_uri
                    )

    def _get_sqla_engine(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        nullpool: bool = True,
        source: Optional[str] = None,
        sqlalchemy_uri: Optional[str] = None,
    ) -> Engine:
        sqlalchemy_url: URL = make_url_safe(sqlalchemy_uri if sqlalchemy_uri else self.sqlalchemy_uri_decrypted)
        self.db_engine_spec.validate_database_uri(sqlalchemy_url)
        extra: Any = self.get_extra()
        params: Dict[str, Any] = extra.get('engine_params', {})
        if nullpool:
            params['poolclass'] = NullPool
        connect_args: Dict[str, Any] = params.get('connect_args', {})
        sqlalchemy_url, connect_args = self.db_engine_spec.adjust_engine_params(
            uri=sqlalchemy_url, connect_args=connect_args, catalog=catalog, schema=schema
        )
        effective_username: Optional[str] = self.get_effective_user(sqlalchemy_url)
        if effective_username and is_feature_enabled('IMPERSONATE_WITH_EMAIL_PREFIX'):
            user = security_manager.find_user(username=effective_username)
            if user and user.email:
                effective_username = user.email.split('@')[0]
        oauth2_config: Optional[OAuth2ClientConfig] = self.get_oauth2_config()
        access_token: Optional[str] = (
            get_oauth2_access_token(oauth2_config, self.id, g.user.id, self.db_engine_spec)
            if oauth2_config and hasattr(g, 'user') and hasattr(g.user, 'id')
            else None
        )
        sqlalchemy_url = self.db_engine_spec.get_url_for_impersonation(
            sqlalchemy_url, self.impersonate_user, effective_username, access_token
        )
        masked_url: URL = self.get_password_masked_url(sqlalchemy_url)
        logger.debug('Database._get_sqla_engine(). Masked URL: %s', str(masked_url))
        if self.impersonate_user:
            args: List[Any] = [connect_args, str(sqlalchemy_url), effective_username, access_token]
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

    def add_database_to_signature(self, func: Callable[..., Any], args: List[Any]) -> List[Any]:
        """
        Examines a function signature looking for a database param.
        """
        sig: Signature = signature(func)
        params = sig.parameters.keys()
        if 'database' in params:
            args.insert(list(params).index('database'), self)
        return args

    @contextmanager
    def get_raw_connection(
        self, catalog: Optional[str] = None, schema: Optional[str] = None, nullpool: bool = True, source: Optional[str] = None
    ) -> Generator[Connection, None, None]:
        with self.get_sqla_engine(catalog=catalog, schema=schema, nullpool=nullpool, source=source) as engine:
            with check_for_oauth2(self):
                with closing(engine.raw_connection()) as conn:
                    for prequery in self.db_engine_spec.get_prequeries(database=self, catalog=catalog, schema=schema):
                        cursor = conn.cursor()
                        cursor.execute(prequery)
                    yield conn

    def get_default_catalog(self) -> Any:
        """
        Return the default configured catalog for the database.
        """
        return self.db_engine_spec.get_default_catalog(self)

    def get_default_schema(self, catalog: Any) -> Any:
        """
        Return the default schema for the database.
        """
        return self.db_engine_spec.get_default_schema(self, catalog)

    def get_default_schema_for_query(self, query: Any) -> Any:
        """
        Return the default schema for a given query.
        """
        return self.db_engine_spec.get_default_schema_for_query(self, query)

    @staticmethod
    def post_process_df(df: pd.DataFrame) -> pd.DataFrame:
        def column_needs_conversion(df_series: pd.Series) -> bool:
            return not df_series.empty and isinstance(df_series, pd.Series) and isinstance(df_series.iloc[0], (list, dict))
        for col, coltype in df.dtypes.to_dict().items():
            if coltype == numpy.object_ and column_needs_conversion(df[col]):
                df[col] = df[col].apply(json.json_dumps_w_dates)
        return df

    @property
    def quote_identifier(self) -> Callable[[str], str]:
        """Add quotes to potential identifier expressions if needed"""
        return self.get_dialect().identifier_preparer.quote

    def get_reserved_words(self) -> Set[str]:
        return self.get_dialect().preparer.reserved_words

    def mutate_sql_based_on_config(self, sql_: str, is_split: bool = False) -> str:
        """
        Mutates the SQL query based on the app configuration.
        """
        sql_mutator = config['SQL_QUERY_MUTATOR']
        if sql_mutator and is_split == config['MUTATE_AFTER_SPLIT']:
            return sql_mutator(sql_, security_manager=security_manager, database=self)
        return sql_

    def get_df(
        self,
        sql: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        mutator: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        sqls: List[str] = self.db_engine_spec.parse_sql(sql)
        with self.get_sqla_engine(catalog=catalog, schema=schema) as engine:
            engine_url: URL = engine.url

        def _log_query(sql_inner: str) -> None:
            if log_query:
                log_query(engine_url, sql_inner, schema, __name__, security_manager)

        with self.get_raw_connection(catalog=catalog, schema=schema) as conn:
            cursor = conn.cursor()
            df: Optional[pd.DataFrame] = None
            for i, sql_ in enumerate(sqls):
                sql_ = self.mutate_sql_based_on_config(sql_, is_split=True)
                _log_query(sql_)
                with event_logger.log_context(action='execute_sql', database=self, object_ref=__name__):
                    self.db_engine_spec.execute(cursor, sql_, self)
                rows = self.fetch_rows(cursor, i == len(sqls) - 1)
                if rows is not None:
                    df = self.load_into_dataframe(cursor.description, rows)
            if mutator and df is not None:
                df = mutator(df)
            if df is None:
                raise Exception("No data returned from query")
            return self.post_process_df(df)

    @event_logger.log_this
    def fetch_rows(self, cursor: Any, last: bool) -> Optional[Any]:
        if not last:
            cursor.fetchall()
            return None
        return self.db_engine_spec.fetch_data(cursor)

    @event_logger.log_this
    def load_into_dataframe(self, description: Any, data: Any) -> pd.DataFrame:
        result_set = SupersetResultSet(data, description, self.db_engine_spec)
        return result_set.to_pandas_df()

    def compile_sqla_query(
        self, qry: Any, catalog: Optional[str] = None, schema: Optional[str] = None, is_virtual: bool = False
    ) -> str:
        with self.get_sqla_engine(catalog=catalog, schema=schema) as engine:
            sql: str = str(qry.compile(engine, compile_kwargs={'literal_binds': True}))
            if engine.dialect.identifier_preparer._double_percents:
                sql = sql.replace('%%', '%')
        if is_feature_enabled('OPTIMIZE_SQL') and is_virtual:
            script = SQLScript(sql, self.db_engine_spec.engine).optimize()
            sql = script.format()
        return sql

    def select_star(
        self,
        table: Table,
        limit: int = 100,
        show_cols: bool = False,
        indent: bool = True,
        latest_partition: bool = False,
        cols: Optional[List[str]] = None,
    ) -> str:
        """Generates a ``select *`` statement in the proper dialect"""
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
            return self.db_engine_spec.select_star(
                self, table, engine=engine, limit=limit, show_cols=show_cols, indent=indent, latest_partition=latest_partition, cols=cols
            )

    def apply_limit_to_sql(self, sql: str, limit: int = 1000, force: bool = False) -> str:
        if self.db_engine_spec.allow_limit_clause:
            return self.db_engine_spec.apply_limit_to_sql(sql, limit, self, force=force)
        return self.db_engine_spec.apply_top_to_sql(sql, limit)

    def safe_sqlalchemy_uri(self) -> str:
        return self.sqlalchemy_uri

    @cache_util.memoized_func(key='db:{self.id}:catalog:{catalog}:schema:{schema}:table_list', cache=cache_manager.cache)
    def get_all_table_names_in_schema(self, catalog: Optional[str] = None, schema: Optional[str] = None) -> Set[Tuple[str, Optional[str], Optional[str]]]:
        """
        :return: The table/schema pairs
        """
        try:
            with self.get_inspector(catalog=catalog, schema=schema) as inspector:
                tables = self.db_engine_spec.get_table_names(database=self, inspector=inspector, schema=schema)
                return {(table, schema, catalog) for table in tables}
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @cache_util.memoized_func(key='db:{self.id}:catalog:{catalog}:schema:{schema}:view_list', cache=cache_manager.cache)
    def get_all_view_names_in_schema(self, catalog: Optional[str] = None, schema: Optional[str] = None) -> Set[Tuple[str, Optional[str], Optional[str]]]:
        """
        :return: set of views
        """
        try:
            with self.get_inspector(catalog=catalog, schema=schema) as inspector:
                views = self.db_engine_spec.get_view_names(database=self, inspector=inspector, schema=schema)
                return {(view, schema, catalog) for view in views}
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @contextmanager
    def get_inspector(
        self, catalog: Optional[str] = None, schema: Optional[str] = None, ssh_tunnel: Optional[Any] = None
    ) -> Generator[Inspector, None, None]:
        with self.get_sqla_engine(catalog=catalog, schema=schema, override_ssh_tunnel=ssh_tunnel) as engine:
            yield sqla.inspect(engine)

    @cache_util.memoized_func(key='db:{self.id}:catalog:{catalog}:schema_list', cache=cache_manager.cache)
    def get_all_schema_names(self, *, catalog: Optional[str] = None, ssh_tunnel: Optional[Any] = None) -> List[str]:
        """
        :return: schema list
        """
        try:
            with self.get_inspector(catalog=catalog, ssh_tunnel=ssh_tunnel) as inspector:
                return self.db_engine_spec.get_schema_names(inspector)
        except Exception as ex:
            if self.is_oauth2_enabled() and self.db_engine_spec.needs_oauth2(ex):
                self.start_oauth2_dance()
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @cache_util.memoized_func(key='db:{self.id}:catalog_list', cache=cache_manager.cache)
    def get_all_catalog_names(self, *, ssh_tunnel: Optional[Any] = None) -> List[str]:
        """
        :return: catalog list
        """
        try:
            with self.get_inspector(ssh_tunnel=ssh_tunnel) as inspector:
                return self.db_engine_spec.get_catalog_names(self, inspector)
        except Exception as ex:
            if self.is_oauth2_enabled() and self.db_engine_spec.needs_oauth2(ex):
                self.start_oauth2_dance()
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @property
    def db_engine_spec(self) -> Any:
        url: URL = make_url_safe(self.sqlalchemy_uri_decrypted)
        return self.get_db_engine_spec(url)

    @classmethod
    @lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
    def get_db_engine_spec(cls, url: URL) -> Any:
        backend: str = url.get_backend_name()
        try:
            driver: Optional[str] = url.get_driver_name()
        except NoSuchModuleError:
            driver = None
        return db_engine_specs.get_engine_spec(backend, driver)

    def grains(self) -> Any:
        """Defines time granularity database-specific expressions."""
        return self.db_engine_spec.get_time_grains()

    def get_extra(self) -> Dict[str, Any]:
        return self.db_engine_spec.get_extra_params(self)

    def get_encrypted_extra(self) -> Dict[str, Any]:
        encrypted_extra: Dict[str, Any] = {}
        if self.encrypted_extra:
            try:
                encrypted_extra = json.loads(self.encrypted_extra)
            except json.JSONDecodeError as ex:
                logger.error(ex, exc_info=True)
                raise
        return encrypted_extra

    def update_params_from_encrypted_extra(self, params: Dict[str, Any]) -> None:
        self.db_engine_spec.update_params_from_encrypted_extra(self, params)

    def get_table(self, table: Table) -> SqlaTable:
        extra = self.get_extra()
        meta = MetaData(**extra.get('metadata_params', {}))
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
            return SqlaTable(table.table, meta, schema=table.schema or None, autoload=True, autoload_with=engine)

    def get_table_comment(self, table: Table) -> Any:
        with self.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
            return self.db_engine_spec.get_table_comment(inspector, table)

    def get_columns(self, table: Table) -> Any:
        with self.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
            return self.db_engine_spec.get_columns(inspector, table, self.schema_options)

    def get_metrics(self, table: Table) -> Any:
        with self.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
            return self.db_engine_spec.get_metrics(self, inspector, table)

    def get_indexes(self, table: Table) -> Any:
        with self.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
            return self.db_engine_spec.get_indexes(self, inspector, table)

    def get_pk_constraint(self, table: Table) -> Dict[str, Any]:
        with self.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
            pk_constraint = inspector.get_pk_constraint(table.table, table.schema) or {}

            def _convert(value: Any) -> Any:
                try:
                    return json.base_json_conv(value)
                except TypeError:
                    return None
            return {key: _convert(value) for key, value in pk_constraint.items()}

    def get_foreign_keys(self, table: Table) -> Any:
        with self.get_inspector(catalog=table.catalog, schema=table.schema) as inspector:
            return inspector.get_foreign_keys(table.table, table.schema)

    def get_schema_access_for_file_upload(self) -> Set[Any]:
        allowed_databases: Any = self.get_extra().get('schemas_allowed_for_file_upload', [])
        if isinstance(allowed_databases, str):
            allowed_databases = literal_eval(allowed_databases)
        if hasattr(g, 'user'):
            extra_allowed_databases = config['ALLOWED_USER_CSV_SCHEMA_FUNC'](self, g.user)
            allowed_databases += extra_allowed_databases
        return set(allowed_databases)

    @property
    def sqlalchemy_uri_decrypted(self) -> str:
        try:
            conn: URL = make_url_safe(self.sqlalchemy_uri)
        except DatabaseInvalidError:
            return 'dialect://invalid_uri'
        if custom_password_store:
            conn = conn.set(password=custom_password_store(conn))
        else:
            conn = conn.set(password=self.password)
        return str(conn)

    @property
    def sql_url(self) -> str:
        return f'/superset/sql/{self.id}/'

    @hybrid_property
    def perm(self) -> str:
        return f'[{self.database_name}].(id:{self.id})'

    @perm.expression
    def perm(cls) -> Any:
        return '[' + cls.database_name + '].(id:' + expression.cast(cls.id, String) + ')'

    def get_perm(self) -> Any:
        return self.perm

    def has_table(self, table: Table) -> bool:
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
            return engine.has_table(table.table, table.schema or None)

    def has_view(self, table: Table) -> bool:
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
            connection = engine.connect()
            try:
                views = engine.dialect.get_view_names(connection=connection, schema=table.schema)
            except Exception:
                logger.warning('Has view failed', exc_info=True)
                views = []
        return table.table in views

    def get_dialect(self) -> Dialect:
        sqla_url: URL = make_url_safe(self.sqlalchemy_uri_decrypted)
        return sqla_url.get_dialect()()

    def make_sqla_column_compatible(self, sqla_col: Any, label: Optional[str] = None) -> Any:
        """Takes a sqlalchemy column object and adds label info if supported by engine."""
        label_expected: str = label or sqla_col.name
        if self.db_engine_spec.get_allows_alias_in_select(self):
            label_adj = self.db_engine_spec.make_label_compatible(label_expected)
            sqla_col = sqla_col.label(label_adj)
        sqla_col.key = label_expected
        return sqla_col

    def is_oauth2_enabled(self) -> bool:
        """
        Is OAuth2 enabled in the database for authentication?
        """
        encrypted_extra: Dict[str, Any] = json.loads(self.encrypted_extra or '{}')
        oauth2_client_info = encrypted_extra.get('oauth2_client_info', {})
        return bool(oauth2_client_info) or self.db_engine_spec.is_oauth2_enabled()

    def get_oauth2_config(self) -> Optional[OAuth2ClientConfig]:
        """
        Return OAuth2 client configuration.
        """
        encrypted_extra: Dict[str, Any] = json.loads(self.encrypted_extra or '{}')
        if (oauth2_client_info := encrypted_extra.get('oauth2_client_info')):
            schema = OAuth2ClientConfigSchema()
            client_config = schema.load(oauth2_client_info)
            return cast(OAuth2ClientConfig, client_config)
        return self.db_engine_spec.get_oauth2_config()

    def start_oauth2_dance(self) -> Any:
        """
        Start the OAuth2 dance.
        """
        return self.db_engine_spec.start_oauth2_dance(self)

    def purge_oauth2_tokens(self) -> None:
        """
        Delete all OAuth2 tokens associated with this database.
        """
        db.session.query(DatabaseUserOAuth2Tokens).filter(DatabaseUserOAuth2Tokens.id == self.id).delete()


sqla.event.listen(Database, 'after_insert', security_manager.database_after_insert)
sqla.event.listen(Database, 'after_update', security_manager.database_after_update)
sqla.event.listen(Database, 'after_delete', security_manager.database_after_delete)


class DatabaseUserOAuth2Tokens(Model, AuditMixinNullable):
    """
    Store OAuth2 tokens, for authenticating to DBs using user personal tokens.
    """
    __tablename__ = 'database_user_oauth2_tokens'
    __table_args__ = (sqla.Index('idx_user_id_database_id', 'user_id', 'database_id'),)
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('ab_user.id', ondelete='CASCADE'), nullable=False)
    user = relationship(security_manager.user_model, foreign_keys=[user_id])
    database_id = Column(Integer, ForeignKey('dbs.id', ondelete='CASCADE'), nullable=False)
    database = relationship('Database', foreign_keys=[database_id])
    access_token = Column(encrypted_field_factory.create(Text), nullable=True)
    access_token_expiration = Column(DateTime, nullable=True)
    refresh_token = Column(encrypted_field_factory.create(Text), nullable=True)


class Log(Model):
    """ORM object used to log Superset actions to the database"""
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True)
    action = Column(String(512))
    user_id = Column(Integer, ForeignKey('ab_user.id'))
    dashboard_id = Column(Integer)
    slice_id = Column(Integer)
    json = Column(utils.MediumText())
    user = relationship(security_manager.user_model, backref='logs', foreign_keys=[user_id])
    dttm = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Integer)
    referrer = Column(String(1024))


class FavStarClassName(StrEnum):
    CHART = 'slice'
    DASHBOARD = 'Dashboard'


class FavStar(UUIDMixin, Model):
    __tablename__ = 'favstar'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('ab_user.id'))
    class_name = Column(String(50))
    obj_id = Column(Integer)
    dttm = Column(DateTime, default=datetime.utcnow)