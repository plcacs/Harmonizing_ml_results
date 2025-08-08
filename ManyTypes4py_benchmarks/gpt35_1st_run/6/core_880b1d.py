from __future__ import annotations
from typing import Any, Callable, cast, TYPE_CHECKING
import sqlalchemy as sqla
from flask_appbuilder import Model
from sqlalchemy import Boolean, Column, create_engine, DateTime, ForeignKey, Integer, MetaData, String, Table as SqlaTable, Text
from sqlalchemy.engine import Connection, Dialect, Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import relationship
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import ColumnElement, expression, Select

if TYPE_CHECKING:
    from superset.databases.ssh_tunnel.models import SSHTunnel
    from superset.models.sql_lab import Query

class KeyValue(Model):
    id: int
    value: str

class CssTemplate(AuditMixinNullable, UUIDMixin, Model):
    id: int
    template_name: str
    css: str

class ConfigurationMethod(StrEnum):
    SQLALCHEMY_FORM: str
    DYNAMIC_FORM: str

class Database(Model, AuditMixinNullable, ImportExportMixin):
    id: int
    verbose_name: str
    database_name: str
    sqlalchemy_uri: str
    password: str
    cache_timeout: int
    select_as_create_table_as: bool
    expose_in_sqllab: bool
    configuration_method: str
    allow_run_async: bool
    allow_file_upload: bool
    allow_ctas: bool
    allow_cvas: bool
    allow_dml: bool
    force_ctas_schema: str
    extra: str
    encrypted_extra: str
    impersonate_user: bool
    server_cert: str
    is_managed_externally: bool
    external_url: str
    export_fields: list[str]
    extra_import_fields: list[str]
    export_children: list[str]

class DatabaseUserOAuth2Tokens(Model, AuditMixinNullable):
    id: int
    user_id: int
    database_id: int
    access_token: str
    access_token_expiration: datetime
    refresh_token: str

class Log(Model):
    id: int
    action: str
    user_id: int
    dashboard_id: int
    slice_id: int
    json: str
    dttm: datetime
    duration_ms: int
    referrer: str

class FavStarClassName(StrEnum):
    CHART: str
    DASHBOARD: str

class FavStar(UUIDMixin, Model):
    id: int
    user_id: int
    class_name: str
    obj_id: int
    dttm: datetime
