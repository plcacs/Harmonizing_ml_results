# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-lines, too-many-arguments

"""A collection of ORM sqlalchemy models for Superset"""

from __future__ import annotations

import builtins
import logging
import textwrap
from ast import literal_eval
from contextlib import closing, contextmanager, nullcontext, suppress
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from inspect import signature
from typing import Any, Callable, cast, TYPE_CHECKING, Optional, List, Set, Tuple, Dict, Union, ContextManager, Iterator

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
custom_password_store = config["SQLALCHEMY_CUSTOM_PASSWORD_STORE"]
stats_logger = config["STATS_LOGGER"]
log_query = config["QUERY_LOGGER"]
metadata = Model.metadata  # pylint: disable=no-member
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from superset.databases.ssh_tunnel.models import SSHTunnel
    from superset.models.sql_lab import Query

DB_CONNECTION_MUTATOR = config["DB_CONNECTION_MUTATOR"]


class KeyValue(Model):  # pylint: disable=too-few-public-methods
    """Used for any type of key-value store"""

    __tablename__ = "keyvalue"
    id: Column[int] = Column(Integer, primary_key=True)
    value: Column[str] = Column(utils.MediumText(), nullable=False)


class CssTemplate(AuditMixinNullable, UUIDMixin, Model):
    """CSS templates for dashboards"""

    __tablename__ = "css_templates"
    id: Column[int] = Column(Integer, primary_key=True)
    template_name: Column[str] = Column(String(250))
    css: Column[str] = Column(utils.MediumText(), default="")


class ConfigurationMethod(StrEnum):
    SQLALCHEMY_FORM = "sqlalchemy_form"
    DYNAMIC_FORM = "dynamic_form"


class Database(Model, AuditMixinNullable, ImportExportMixin):  # pylint: disable=too-many-public-methods
    """An ORM object that stores Database related information"""

    __tablename__ = "dbs"
    type: str = "table"
    __table_args__ = (UniqueConstraint("database_name"),)

    id: Column[int] = Column(Integer, primary_key=True)
    verbose_name: Column[str] = Column(String(250), unique=True)
    # short unique name, used in permissions
    database_name: Column[str] = Column(String(250), unique=True, nullable=False)
    sqlalchemy_uri: Column[str] = Column(String(1024), nullable=False)
    password: Column[Optional[str]] = Column(encrypted_field_factory.create(String(1024)))
    cache_timeout: Column[Optional[int]] = Column(Integer)
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
    force_ctas_schema: Column[Optional[str]] = Column(String(250))
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
    encrypted_extra: Column[Optional[str]] = Column(encrypted_field_factory.create(Text), nullable=True)
    impersonate_user: Column[bool] = Column(Boolean, default=False)
    server_cert: Column[Optional[str]] = Column(encrypted_field_factory.create(Text), nullable=True)
    is_managed_externally: Column[bool] = Column(Boolean, nullable=False, default=False)
    external_url: Column[Optional[str]] = Column(Text, nullable=True)

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
        extra = self.get_extra() or {}
        cost_estimate_enabled: bool = extra.get("cost_estimate_enabled", False)
        return (
            self.db_engine_spec.get_allow_cost_estimate(extra) and cost_estimate_enabled
        )

    @property
    def allows_virtual_table_explore(self) -> bool:
        extra = self.get_extra() or {}
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
        masked_uri = make_url_safe(self.sqlalchemy_uri)
        encrypted_config: Dict[str, Any] = {}
        if (masked_encrypted_extra := self.masked_encrypted_extra) is not None:
            with suppress(TypeError, json.JSONDecodeError):
                encrypted_config = json.loads(masked_encrypted_extra)
        try:
            parameters = self.db_engine_spec.get_parameters_from_uri(
                masked_uri,
                encrypted_extra=encrypted_config,
            )
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

    def set_sqlalchemy_uri(self, uri: str) -> None:
        conn = make_url_safe(uri.strip())
        if conn.password != PASSWORD_MASK and not custom_password_store:
            self.password = conn.password
        conn = conn.set(password=PASSWORD_MASK if conn.password else None)
        self.sqlalchemy_uri = str(conn)

    def get_effective_user(self, object_url: URL) -> Optional[str]:
        return (
            username
            if (username := get_username())
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
        override_ssh_tunnel: Optional["SSHTunnel"] = None,
    ) -> Iterator[Engine]:
        from superset.daos.database import DatabaseDAO

        sqlalchemy_uri = self.sqlalchemy_uri_decrypted

        ssh_tunnel = override_ssh_tunnel or DatabaseDAO.get_ssh_tunnel(self.id)
        ssh_context_manager = (
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

            engine_context_manager = config["ENGINE_CONTEXT_MANAGER"]
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
        sqlalchemy_url = make_url_safe(
            sqlalchemy_uri if sqlalchemy_uri else self.sqlalchemy_uri_decrypted
        )
        self.db_engine_spec.validate_database_uri(sqlalchemy_url)

        extra = self.get_extra()
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

        effective_username = self.get_effective_user(sqlalchemy_url)
        if effective_username and is_feature_enabled("IMPERSONATE_WITH_EMAIL_PREFIX"):
            user = security_manager.find_user(username=effective_username)
            if user and user.email:
                effective_username = user.email.split("@")[0]

        oauth2_config = self.get_oauth2_config()
        access_token = (
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
