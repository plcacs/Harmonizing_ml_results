from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast
from datetime import datetime
from functools import lru_cache
from inspect import signature
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from ast import literal_eval
import logging
import textwrap
import builtins
import numpy
import pandas as pd
import sqlalchemy as sqla
import sshtunnel
from flask import g, request
from flask_appbuilder import Model
from sqlalchemy import (
    Boolean,
    Column,
    create_engine,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table as SqlaTable,
    Text,
)
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
from superset.extensions import (
    cache_manager,
    encrypted_field_factory,
    event_logger,
    security_manager,
    ssh_manager_factory,
)
from superset.models.helpers import AuditMixinNullable, ImportExportMixin, UUIDMixin
from superset.result_set import SupersetResultSet
from superset.sql.parse import SQLScript
from superset.sql_parse import Table
from superset.superset_typing import (
    DbapiDescription,
    OAuth2ClientConfig,
    ResultSetColumnType,
)
from superset.utils import cache as cache_util, core as utils, json
from superset.utils.backports import StrEnum
from superset.utils.core import get_username
from superset.utils.oauth2 import (
    check_for_oauth2,
    get_oauth2_access_token,
    OAuth2ClientConfigSchema,
)

config = app.config
custom_password_store: Callable[[URL], str] = config["SQLALCHEMY_CUSTOM_PASSWORD_STORE"]
stats_logger: Callable[[str, str, Optional[str], str, Any], None] = config["STATS_LOGGER"]
log_query: Callable[[str, str, Optional[str], None] = config["QUERY_LOGGER"]
metadata: MetaData = Model.metadata  # pylint: disable=no-member
logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from superset.databases.ssh_tunnel.models import SSHTunnel
    from superset.models.sql_lab import Query

DB_CONNECTION_MUTATOR: Callable[[URL, Dict[str, Any], Tuple[URL, Dict[str, Any]]] = config["DB_CONNECTION_MUTATOR"]


class KeyValue(Model):  # pylint: disable=too-few-public-methods
    """Used for any type of key-value store"""

    __tablename__: str = "keyvalue"
    id: Column[int] = Column(Integer, primary_key=True)
    value: Column[str] = Column(utils.MediumText(), nullable=False)


class CssTemplate(AuditMixinNullable, UUIDMixin, Model):
    """CSS templates for dashboards"""

    __tablename__: str = "css_templates"
    id: Column[int] = Column(Integer, primary_key=True)
    template_name: Column[str] = Column(String(250))
    css: Column[str] = Column(utils.MediumText(), default="")


class ConfigurationMethod(StrEnum):
    SQLALCHEMY_FORM: str = "sqlalchemy_form"
    DYNAMIC_FORM: str = "dynamic_form"


class Database(Model, AuditMixinNullable, ImportExportMixin):  # pylint: disable=too-many-public-methods
    """An ORM object that stores Database related information"""

    __tablename__: str = "dbs"
    type: str = "table"
    __table_args__: Tuple[UniqueConstraint] = (UniqueConstraint("database_name"),)

    id: Column[int] = Column(Integer, primary_key=True)
    verbose_name: Column[str] = Column(String(250), unique=True)
    database_name: Column[str] = Column(String(250), unique=True, nullable=False)
    sqlalchemy_uri: Column[str] = Column(String(1024), nullable=False)
    password: Column[str] = Column(encrypted_field_factory.create(String(1024)))
    cache_timeout: Column[int] = Column(Integer)
    select_as_create_table_as: Column[bool] = Column(Boolean, default=False)
    expose_in_sqllab: Column[bool] = Column(Boolean, default=True)
    configuration_method: Column[str] = Column(
        String(255), server_default=ConfigurationMethod.SQLALCHEMY_FORM.value
    )
    allow_run_async: Column[bool] = Column(Boolean, default=False)
    allow_file_upload: Column[bool] = Column(Boolean, default=False)
    allow_ctas: Column[bool] = Column(Boolean, default=False)
    allow_cvas: Column[bool] = Column(Boolean, default=False)
    allow_dml: Column[bool] = Column(Boolean, default=False)
    force_ctas_schema: Column[str] = Column(String(250))
    extra: Column[str] = Column(
        Text,
        default=textwrap.dedent(
            """\
    {
        "metadata_params": {},
        "engine_params": {},
        "metadata_cache_timeout": {},
        "schemas_allowed_for_file_upload": []
    }
    """
        ),
    )
    encrypted_extra: Column[str] = Column(encrypted_field_factory.create(Text), nullable=True)
    impersonate_user: Column[bool] = Column(Boolean, default=False)
    server_cert: Column[str] = Column(encrypted_field_factory.create(Text), nullable=True)
    is_managed_externally: Column[bool] = Column(Boolean, nullable=False, default=False)
    external_url: Column[str] = Column(Text, nullable=True)

    export_fields: List[str] = [
        "database_name",
        "sqlalchemy_uri",
        "cache_timeout",
        "expose_in_sqllab",
        "allow_run_async",
        "allow_ctas",
        "allow_cvas",
        "allow_dml",
        "allow_file_upload",
        "extra",
        "impersonate_user",
    ]
    extra_import_fields: List[str] = [
        "password",
        "is_managed_externally",
        "external_url",
        "encrypted_extra",
        "impersonate_user",
    ]
    export_children: List[str] = ["tables"]

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self.verbose_name if self.verbose_name else self.database_name

    @property
    def allows_subquery(self) -> bool:
        return self.db_engine_spec.allows_subqueries

    @property
    def function_names(self) -> List[str]:
        try:
            return self.db_engine_spec.get_function_names(self)
        except Exception as ex:  # pylint: disable=broad-except
            logger.error(
                "Failed to fetch database function names with error: %s",
                str(ex),
                exc_info=True,
            )
        return []

    @property
    def allows_cost_estimate(self) -> bool:
        extra: Dict[str, Any] = self.get_extra() or {}
        cost_estimate_enabled: bool = extra.get("cost_estimate_enabled")  # type: ignore

        return (
            self.db_engine_spec.get_allow_cost_estimate(extra) and cost_estimate_enabled
        )

    @property
    def allows_virtual_table_explore(self) -> bool:
        extra: Dict[str, Any] = self.get_extra()
        return bool(extra.get("allows_virtual_table_explore", True))

    @property
    def explore_database_id(self) -> int:
        return self.get_extra().get("explore_database_id", self.id)

    @property
    def disable_data_preview(self) -> bool:
        return self.get_extra().get("disable_data_preview", False) is True

    @property
    def disable_drill_to_detail(self) -> bool:
        return self.get_extra().get("disable_drill_to_detail", False) is True

    @property
    def allow_multi_catalog(self) -> bool:
        return self.get_extra().get("allow_multi_catalog", False)

    @property
    def schema_options(self) -> Dict[str, Any]:
        return self.get_extra().get("schema_options", {})

    @property
    def data(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.database_name,
            "backend": self.backend,
            "configuration_method": self.configuration_method,
            "allows_subquery": self.allows_subquery,
            "allows_cost_estimate": self.allows_cost_estimate,
            "allows_virtual_table_explore": self.allows_virtual_table_explore,
            "explore_database_id": self.explore_database_id,
            "schema_options": self.schema_options,
            "parameters": self.parameters,
            "disable_data_preview": self.disable_data_preview,
            "disable_drill_to_detail": self.disable_drill_to_detail,
            "allow_multi_catalog": self.allow_multi_catalog,
            "parameters_schema": self.parameters_schema,
            "engine_information": self.engine_information,
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
            parameters: Dict[str, Any] = self.db_engine_spec.get_parameters_from_uri(  # type: ignore
                masked_uri,
                encrypted_extra=encrypted_config,
            )
        except Exception:  # pylint: disable=broad-except
            parameters = {}

        return parameters

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        try:
            parameters_schema: Dict[str, Any] = self.db_engine_spec.parameters_json_schema()  # type: ignore
        except Exception:  # pylint: disable=broad-except
            parameters_schema = {}
        return parameters_schema

    @property
    def metadata_cache_timeout(self) -> Dict[str, Any]:
        return self.get_extra().get("metadata_cache_timeout", {})

    @property
    def catalog_cache_enabled(self) -> bool:
        return "catalog_cache_timeout" in self.metadata_cache_timeout

    @property
    def catalog_cache_timeout(self) -> Optional[int]:
        return self.metadata_cache_timeout.get("catalog_cache_timeout")

    @property
    def schema_cache_enabled(self) -> bool:
        return "schema_cache_timeout" in self.metadata_cache_timeout

    @property
    def schema_cache_timeout(self) -> Optional[int]:
        return self.metadata_cache_timeout.get("schema_cache_timeout")

    @property
    def table_cache_enabled(self) -> bool:
        return "table_cache_timeout" in self.metadata_cache_timeout

    @property
    def table_cache_timeout(self) -> Optional[int]:
        return self.metadata_cache_timeout.get("table_cache_timeout")

    @property
    def default_schemas(self) -> List[str]:
        return self.get_extra().get("default_schemas", [])

    @property
    def connect_args(self) -> Dict[str, Any]:
        return self.get_extra().get("engine_params", {}).get("connect_args", {})

    @property
    def engine_information(self) -> Dict[str, Any]:
        try:
            engine_information: Dict[str, Any] = self.db_engine_spec.get_public_information()
        except Exception:  # pylint: disable=broad-except
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
        if conn.password != PASSWORD_MASK and not custom_password_store:
            self.password = conn.password
        conn = conn.set(password=PASSWORD_MASK if conn.password else None)
        self.sqlalchemy_uri = str(conn)

    def get_effective_user(self, object_url: URL) -> Optional[str]:
        username: Optional[str] = get_username()
        return (
            username
            if username
            else object_url.username
            if self.impersonate_user
            else None
        )

    @contextmanager
    def get_sqla_engine(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        nullpool: bool = True,
        source: Optional[utils.QuerySource] = None,
        override_ssh_tunnel: Optional[SSHTunnel] = None,
    ) -> Engine:
        from superset.daos.database import DatabaseDAO

        sqlalchemy_uri: str = self.sqlalchemy_uri_decrypted

        ssh_tunnel: Optional[SSHTunnel] = override_ssh_tunnel or DatabaseDAO.get_ssh_tunnel(self.id)
        ssh_context_manager: Any = (
            ssh_manager_factory.instance.create_tunnel(
                ssh_tunnel=ssh_tunnel,
                sqlalchemy_database_uri=sqlalchemy_uri,
            )
            if ssh_tunnel
            else nullcontext()
        )

        with ssh_context_manager as ssh_context:
            if ssh_context:
                logger.info(
                    "[SSH] Successfully created tunnel w/ %s tunnel_timeout + %s "
                    "ssh_timeout at %s",
                    sshtunnel.TUNNEL_TIMEOUT,
                    sshtunnel.SSH_TIMEOUT,
                    ssh_context.local_bind_address,
                )
                sqlalchemy_uri = ssh_manager_factory.instance.build_sqla_url(
                    sqlalchemy_uri,
                    ssh_context,
                )

            engine_context_manager: Any = config["ENGINE_CONTEXT_MANAGER"]
            with engine_context_manager(self, catalog, schema):
                with check_for_oauth2(self):
                    yield self._get_sqla_engine(
                        catalog=catalog,
                        schema=schema,
                        nullpool=nullpool,
                        source=source,
                        sqlalchemy_uri=sqlalchemy_uri,
                    )

    def _get_sqla_engine(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        nullpool: bool = True,
        source: Optional[utils.QuerySource] = None,
        sqlalchemy_uri: Optional[str] = None,
    ) -> Engine:
        sqlalchemy_url: URL = make_url_safe(
            sqlalchemy_uri if sqlalchemy_uri else self.sqlalchemy_uri_decrypted
        )
        self.db_engine_spec.validate_database_uri(sqlalchemy_url)

        extra: Dict[str, Any] = self.get_extra()
        params: Dict[str, Any] = extra.get("engine_params", {})
        if nullpool:
            params["poolclass"] = NullPool
        connect_args: Dict[str, Any] = params.get("connect_args", {})

        sqlalchemy_url, connect_args = self.db_engine_spec.adjust_engine_params(
            uri=sqlalchemy_url,
            connect_args=connect_args,
            catalog=catalog,
            schema=schema,
        )

        effective_username: Optional[str] = self.get_effective_user(sqlalchemy_url)
        if effective_username and is_feature_enabled("IMPERSONATE_WITH_EMAIL_PREFIX"):
            user: Any = security_manager.find_user(username=effective_username)
            if user and user.email:
                effective_username = user.email.split("@")[0]

        oauth2_config: Optional[OAuth2ClientConfig] = self.get_oauth2_config()
        access_token: Optional[str] = (
            get_oauth2_access_token(
                oauth2_config,
                self.id,
                g.user.id,
                self.db_engine_spec,
            )
            if oauth2_config and hasattr(g, "user") and hasattr(g.user, "id")
            else None
        )
        sqlalchemy_url = self.db_engine_spec.get_url_for_impersonation(
            sqlalchemy_url,
            self.impersonate_user,
            effective_username,
            access_token,
        )

        masked_url: URL = self.get_password_masked_url(sqlalchemy_url)
        logger.debug("Database._get_sqla_engine(). Masked URL: %s", str(masked_url))

        if self.impersonate_user:
            args: List[Any] = [connect_args, str(sqlalchemy_url), effective_username, access_token]
            args = self.add_database_to_signature(
                self.db_engine_spec.update_impersonation_config,
                args,
            )
            self.db_engine_spec.update_impersonation_config(*args)

        if connect_args:
            params["connect_args"] = connect_args

        self.update_params_from_encrypted_extra(params)

        if DB_CONNECTION_MUTATOR:
            if not source and request and request.referrer:
                if "/superset/dashboard/" in request.referrer:
                    source = utils.QuerySource.DASHBOARD
                elif "/explore/" in request.referrer:
                    source = utils.QuerySource.CHART
                elif "/sqllab/" in request.referrer:
                    source = utils.QuerySource.SQL_LAB

            sqlalchemy_url, params = DB_CONNECTION_MUTATOR(
                sqlalchemy_url,
                params,
                effective_username,
                security_manager,
                source,
            )
        try:
            return create_engine(sqlalchemy_url, **params)
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    def add_database_to_signature(
        self,
        func: Callable[..., None],
        args: List[Any],
    ) -> List[Any]:
        sig: Any = signature(func)
        if "database" in (params := sig.parameters.keys()):
            args.insert(list(params).index("database"), self)
        return args

    @contextmanager
    def get_raw_connection(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        nullpool: bool = True,
        source: Optional[utils.QuerySource] = None,
    ) -> Connection:
        with self.get_sqla_engine(
            catalog=catalog,
            schema=schema,
            nullpool=nullpool,
            source=source,
        ) as engine:
            with check_for_oauth2(self):
                with closing(engine.raw_connection()) as conn:
                    for prequery in self.db_engine_spec.get_prequeries(
                        database=self,
                        catalog=catalog,
                        schema=schema,
                    ):
                        cursor: Any = conn.cursor()
                        cursor.execute(prequery)

                    yield conn

    def get_default_catalog(self) -> Optional[str]:
        return self.db_engine_spec.get_default_catalog(self)

    def get_default_schema(self, catalog: Optional[str]) -> Optional[str]:
        return self.db_engine_spec.get_default_schema(self, catalog)

    def get_default_schema_for_query(self, query: Query) -> Optional[str]:
        return self.db_engine_spec.get_default_schema_for_query(self, query)

    @staticmethod
    def post_process_df(df: pd.DataFrame) -> pd.DataFrame:
        def column_needs_conversion(df_series: pd.Series) -> bool:
            return (
                not df_series.empty
                and isinstance(df_series, pd.Series)
                and isinstance(df_series[0], (list, dict))
            )

        for col, coltype in df.dtypes.to_dict().items():
            if coltype == numpy.object_ and column_needs_conversion(df[col]):
                df[col] = df[col].apply(json.json_dumps_w_dates)
        return df

    @property
    def quote_identifier(self) -> Callable[[str], str]:
        return self.get_dialect().identifier_preparer.quote

    def get_reserved_words(self) -> Set[str]:
        return self.get_dialect().preparer.reserved_words

    def mutate_sql_based_on_config(self, sql_: str, is_split: bool = False) -> str:
        sql_mutator: Optional[Callable[[str, Any, Any], str]] = config["SQL_QUERY_MUTATOR"]
        if sql_mutator and (is_split == config["MUTATE_AFTER_SPLIT"]):
            return sql_mutator(
                sql_,
                security_manager=security_manager,
                database=self,
            )
        return sql_

    def get_df(
        self,
        sql: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        mutator: Optional[Callable[[pd.DataFrame], None]] = None,
    ) -> pd.DataFrame:
        sqls: List[str] = self.db_engine_spec.parse_sql(sql)
        with self.get_sqla_engine(catalog=catalog, schema=schema) as engine:
            engine_url: URL = engine.url

        def _log_query(sql: str) -> None:
            if log_query:
                log_query(
                    engine_url,
                    sql,
                    schema,
                    __name__,
                    security_manager,
                )

        with self.get_raw_connection(catalog=catalog, schema=schema) as conn:
            cursor: Any = conn.cursor()
            df: Optional[pd.DataFrame] = None
            for i, sql_ in enumerate(sqls):
                sql_ = self.mutate_sql_based_on_config(sql_, is_split=True)
                _log_query(sql_)
                with event_logger.log_context(
                    action="execute_sql",
                    database=self,
                    object_ref=__name__,
                ):
                    self.db_engine_spec.execute(cursor, sql_, self)

                rows: Optional[List[Tuple[Any, ...]]] = self.fetch_rows(cursor, i == len(sqls) - 1)
                if rows is not None:
                    df = self.load_into_dataframe(cursor.description, rows)

            if mutator:
                df = mutator(df)

            return self.post_process_df(df)

    @event_logger.log_this
    def fetch_rows(self, cursor: Any, last: bool) -> Optional[List[Tuple[Any, ...]]]:
        if not last:
            cursor.fetchall()
            return None

        return self.db_engine_spec.fetch_data(cursor)

    @event_logger.log_this
    def load_into_dataframe(
        self,
        description: DbapiDescription,
        data: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        result_set: SupersetResultSet = SupersetResultSet(
            data,
            description,
            self.db_engine_spec,
        )
        return result_set.to_pandas_df()

    def compile_sqla_query(
        self,
        qry: Select,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        is_virtual: bool = False,
    ) -> str:
        with self.get_sqla_engine(catalog=catalog, schema=schema) as engine:
            sql: str = str(qry.compile(engine, compile_kwargs={"literal_binds": True}))

            if engine.dialect.identifier_preparer._double_percents:  # noqa
                sql = sql.replace("%%", "%")

        if is_feature_enabled("OPTIMIZE_SQL") and is_virtual:
            script: SQLScript = SQLScript(sql, self.db_engine_spec.engine).optimize()
            sql = script.format()

        return sql

    def select_star(
        self,
        table: Table,
        limit: int = 100,
        show_cols: bool = False,
        indent: bool = True,
        latest_partition: bool = False,
        cols: Optional[List[ResultSetColumnType]] = None,
    ) -> str:
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
            return self.db_engine_spec.select_star(
                self,
                table,
                engine=engine,
                limit=limit,
                show_cols=show_cols,
                indent=indent,
                latest_partition=latest_partition,
                cols=cols,
            )

    def apply_limit_to_sql(
        self, sql: str, limit: int = 1000, force: bool = False
    ) -> str:
        if self.db_engine_spec.allow_limit_clause:
            return self.db_engine_spec.apply_limit_to_sql(sql, limit, self, force=force)
        return self.db_engine_spec.apply_top_to_sql(sql, limit)

    def safe_sqlalchemy_uri(self) -> str:
        return self.sqlalchemy_uri

    @cache_util.memoized_func(
        key="db:{self.id}:catalog:{catalog}:schema:{schema}:table_list",
        cache=cache_manager.cache,
    )
    def get_all_table_names_in_schema(
        self,
        catalog: Optional[str],
        schema: str,
    ) -> Set[Tuple[str, str, Optional[str]]]:
        try:
            with self.get_inspector(catalog=catalog, schema=schema) as inspector:
                return {
                    (table, schema, catalog)
                    for table in self.db_engine_spec.get_table_names(
                        database=self,
                        inspector=inspector,
                        schema=schema,
                    )
                }
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @cache_util.memoized_func(
        key="db:{self.id}:catalog:{catalog}:schema:{schema}:view_list",
        cache=cache_manager.cache,
    )
    def get_all_view_names_in_schema(
        self,
        catalog: Optional[str],
        schema: str,
    ) -> Set[Tuple[str, str, Optional[str]]]:
        try:
            with self.get_inspector(catalog=catalog, schema=schema) as inspector:
                return {
                    (view, schema, catalog)
                    for view in self.db_engine_spec.get_view_names(
                        database=self,
                        inspector=inspector,
                        schema=schema,
                    )
                }
        except Exception as ex:
            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @contextmanager
    def get_inspector(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        ssh_tunnel: Optional[SSHTunnel] = None,
    ) -> Inspector:
        with self.get_sqla_engine(
            catalog=catalog,
            schema=schema,
            override_ssh_tunnel=ssh_tunnel,
        ) as engine:
            yield sqla.inspect(engine)

    @cache_util.memoized_func(
        key="db:{self.id}:catalog:{catalog}:schema_list",
        cache=cache_manager.cache,
    )
    def get_all_schema_names(
        self,
        *,
        catalog: Optional[str] = None,
        ssh_tunnel: Optional[SSHTunnel] = None,
    ) -> Set[str]:
        try:
            with self.get_inspector(
                catalog=catalog,
                ssh_tunnel=ssh_tunnel,
            ) as inspector:
                return self.db_engine_spec.get_schema_names(inspector)
        except Exception as ex:
            if self.is_oauth2_enabled() and self.db_engine_spec.needs_oauth2(ex):
                self.start_oauth2_dance()

            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @cache_util.memoized_func(
        key="db:{self.id}:catalog_list",
        cache=cache_manager.cache,
    )
    def get_all_catalog_names(
        self,
        *,
        ssh_tunnel: Optional[SSHTunnel] = None,
    ) -> Set[str]:
        try:
            with self.get_inspector(ssh_tunnel=ssh_tunnel) as inspector:
                return self.db_engine_spec.get_catalog_names(self, inspector)
        except Exception as ex:
            if self.is_oauth2_enabled() and self.db_engine_spec.needs_oauth2(ex):
                self.start_oauth2_dance()

            raise self.db_engine_spec.get_dbapi_mapped_exception(ex) from ex

    @property
    def db_engine_spec(self) -> Type[db_engine_specs.BaseEngineSpec]:
        url: URL = make_url_safe(self.sqlalchemy_uri_decrypted)
        return self.get_db_engine_spec(url)

    @classmethod
    @lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
    def get_db_engine_spec(
        cls, url: URL
    ) -> Type[db_engine_specs.BaseEngineSpec]:
        backend: str = url.get_backend_name()
        try:
            driver: Optional[str] = url.get_driver_name()
        except NoSuchModuleError:
            driver = None

        return db_engine_specs.get_engine_spec(backend, driver)

    def grains(self) -> Tuple[TimeGrain, ...]:
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
        extra: Dict[str, Any] = self.get_extra()
        meta: MetaData = MetaData(**extra.get("metadata_params", {}))
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
            return SqlaTable(
                table.table,
                meta,
                schema=table.schema or None,
                autoload=True,
                autoload_with=engine,
            )

    def get_table_comment(self, table: Table) -> Optional[str]:
        with self.get_inspector(
            catalog=table.catalog,
            schema=table.schema,
        ) as inspector:
            return self.db_engine_spec.get_table_comment(inspector, table)

    def get_columns(self, table: Table) -> List[ResultSetColumnType]:
        with self.get_inspector(
            catalog=table.catalog,
            schema=table.schema,
        ) as inspector:
            return self.db_engine_spec.get_columns(
                inspector, table, self.schema_options
            )

    def get_metrics(
        self,
        table: Table,
    ) -> List[MetricType]:
        with self.get_inspector(
            catalog=table.catalog,
            schema=table.schema,
        ) as inspector:
            return self.db_engine_spec.get_metrics(self, inspector, table)

    def get_indexes(self, table: Table) -> List[Dict[str, Any]]:
        with self.get_inspector(
            catalog=table.catalog,
            schema=table.schema,
        ) as inspector:
            return self.db_engine_spec.get_indexes(self, inspector, table)

    def get_pk_constraint(self, table: Table) -> Dict[str, Any]:
        with self.get_inspector(
            catalog=table.catalog,
            schema=table.schema,
        ) as inspector:
            pk_constraint: Dict[str, Any] = inspector.get_pk_constraint(table.table, table.schema) or {}

            def _convert(value: Any) -> Any:
                try:
                    return json.base_json_conv(value)
                except TypeError:
                    return None

            return {key: _convert(value) for key, value in pk_constraint.items()}

    def get_foreign_keys(self, table: Table) -> List[Dict[str, Any]]:
        with self.get_inspector(
            catalog=table.catalog,
            schema=table.schema,
        ) as inspector:
            return inspector.get_foreign_keys(table.table, table.schema)

    def get_schema_access_for_file_upload(self) -> Set[str]:
        allowed_databases: Union[List[str], str] = self.get_extra().get("schemas_allowed_for_file_upload", [])

        if isinstance(allowed_databases, str):
            allowed_databases = literal_eval(allowed_databases)

        if hasattr(g, "user"):
            extra_allowed_databases: List[str] = config["ALLOWED_USER_CSV_SCHEMA_FUNC"](
                self, g.user
            )
            allowed_databases += extra_allowed_databases
        return set(allowed_databases)

    @property
    def sqlalchemy_uri_decrypted(self) -> str:
        try:
            conn: URL = make_url_safe(self.sqlalchemy_uri)
        except DatabaseInvalidError:
            return "dialect://invalid_uri"
        if custom_password_store:
            conn = conn.set(password=custom_password_store(conn))
        else:
            conn = conn.set(password=self.password)
        return str(conn)

    @property
    def sql_url(self) -> str:
        return f"/superset/sql/{self.id}/"

    @hybrid_property
    def perm(self) -> str:
        return f"[{self.database_name}].(id:{self.id})"

    @perm.expression
    def perm(cls) -> str:  # pylint: disable=no-self-argument
        return (
            "[" + cls.database_name + "].(id:" + expression.cast(cls.id, String) + ")"
        )

    def get_perm(self) -> str:
        return self.perm

    def has_table(self, table: Table) -> bool:
        with self.get_sqla_engine(catalog=table.catalog, schema=table.schema) as engine:
