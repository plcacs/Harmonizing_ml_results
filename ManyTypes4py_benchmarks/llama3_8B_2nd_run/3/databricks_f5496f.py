from __future__ import annotations
from datetime import datetime
from typing import Any, TYPE_CHECKING, TypedDict, Union
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from flask_babel import gettext as __
from marshmallow import fields, Schema
from marshmallow.validate import Range
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from superset.constants import TimeGrain, USER_AGENT
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.base import BaseEngineSpec, BasicParametersMixin
from superset.db_engine_specs.hive import HiveEngineSpec
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.utils import json
from superset.utils.network import is_hostname_valid, is_port_open

if TYPE_CHECKING:
    from superset.models.core import Database

class DatabricksBaseSchema(Schema):
    """
    Fields that are required for both Databricks drivers that uses a
    dynamic form.
    """
    access_token: str
    host: str
    port: int
    encryption: bool

class DatabricksBaseParametersType(TypedDict):
    """
    The parameters are all the keys that do not exist on the Database model.
    These are used to build the sqlalchemy uri.
    """

class DatabricksNativeSchema(DatabricksBaseSchema):
    """
    Additional fields required only for the DatabricksNativeEngineSpec.
    """
    database: str

class DatabricksNativePropertiesSchema(DatabricksNativeSchema):
    """
    Properties required only for the DatabricksNativeEngineSpec.
    """
    http_path: str

class DatabricksNativeParametersType(DatabricksBaseParametersType):
    """
    Additional parameters required only for the DatabricksNativeEngineSpec.
    """

class DatabricksNativePropertiesType(TypedDict):
    """
    All properties that need to be available to the DatabricksNativeEngineSpec
    in order to create a connection if the dynamic form is used.
    """

class DatabricksPythonConnectorSchema(DatabricksBaseSchema):
    """
    Additional fields required only for the DatabricksPythonConnectorEngineSpec.
    """
    http_path_field: str
    default_catalog: str
    default_schema: str

class DatabricksPythonConnectorParametersType(DatabricksBaseParametersType):
    """
    Additional parameters required only for the DatabricksPythonConnectorEngineSpec.
    """

class DatabricksPythonConnectorPropertiesType(TypedDict):
    """
    All properties that need to be available to the DatabricksPythonConnectorEngineSpec
    in order to create a connection if the dynamic form is used.
    """
time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "date_trunc('second', {col})", TimeGrain.MINUTE: "date_trunc('minute', {col})", TimeGrain.HOUR: "date_trunc('hour', {col})", TimeGrain.DAY: "date_trunc('day', {col})", TimeGrain.WEEK: "date_trunc('week', {col})", TimeGrain.MONTH: "date_trunc('month', {col})", TimeGrain.QUARTER: "date_trunc('quarter', {col})", TimeGrain.YEAR: "date_trunc('year', {col})", TimeGrain.WEEK_ENDING_SATURDAY: "date_trunc('week', {col} + interval '1 day') + interval '5 days'", TimeGrain.WEEK_STARTING_SUNDAY: "date_trunc('week', {col} + interval '1 day') - interval '1 day'"}

class DatabricksHiveEngineSpec(HiveEngineSpec):
    engine_name: str
    engine: str
    drivers: dict[str, str]
    default_driver: str
    _show_functions_column: str
    _time_grain_expressions: dict[str, str]

class DatabricksBaseEngineSpec(BaseEngineSpec):
    _time_grain_expressions: dict[str, str]

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: dict[str, Any] = None) -> datetime:
        return HiveEngineSpec.convert_dttm(target_type, dttm, db_extra=db_extra)

    @classmethod
    def epoch_to_dttm(cls) -> datetime:
        return HiveEngineSpec.epoch_to_dttm()

class DatabricksODBCEngineSpec(DatabricksBaseEngineSpec):
    engine_name: str
    engine: str
    drivers: dict[str, str]
    default_driver: str

class DatabricksDynamicBaseEngineSpec(BasicParametersMixin, DatabricksBaseEngineSpec):
    default_driver: str
    encryption_parameters: dict[str, str]
    required_parameters: set[str]
    context_key_mapping: dict[str, str]

    @staticmethod
    def get_extra_params(database: Database) -> dict[str, Any]:
        # ...

    @classmethod
    def get_table_names(cls, database: Database, inspector: Inspector, schema: str) -> set[str]:
        # ...

    @classmethod
    def extract_errors(cls, ex: Exception, context: dict[str, Any] = None) -> list[SupersetError]:
        # ...

    @classmethod
    def validate_parameters(cls, properties: DatabricksNativePropertiesType) -> list[SupersetError]:
        # ...

class DatabricksNativeEngineSpec(DatabricksDynamicBaseEngineSpec):
    engine: str
    engine_name: str
    drivers: dict[str, str]
    default_driver: str
    parameters_schema: DatabricksNativeSchema
    properties_schema: DatabricksNativePropertiesSchema
    sqlalchemy_uri_placeholder: str
    context_key_mapping: dict[str, str]
    required_parameters: set[str]
    supports_dynamic_schema: bool
    supports_catalog: bool
    supports_dynamic_catalog: bool

    @classmethod
    def build_sqlalchemy_uri(cls, parameters: DatabricksNativeParametersType, *_: Any) -> URL:
        # ...

    @classmethod
    def get_parameters_from_uri(cls, uri: str, *_: Any, **__: Any) -> dict[str, Any]:
        # ...

    @classmethod
    def parameters_json_schema(cls) -> dict[str, Any]:
        # ...

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        # ...

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> set[str]:
        # ...

class DatabricksPythonConnectorEngineSpec(DatabricksDynamicBaseEngineSpec):
    engine: str
    engine_name: str
    default_driver: str
    drivers: dict[str, str]
    parameters_schema: DatabricksPythonConnectorSchema
    sqlalchemy_uri_placeholder: str
    context_key_mapping: dict[str, str]
    required_parameters: set[str]
    supports_dynamic_schema: bool
    supports_catalog: bool
    supports_dynamic_catalog: bool

    @classmethod
    def build_sqlalchemy_uri(cls, parameters: DatabricksPythonConnectorParametersType, *_: Any) -> URL:
        # ...

    @classmethod
    def get_parameters_from_uri(cls, uri: str, *_: Any, **__: Any) -> dict[str, Any]:
        # ...

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        # ...

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> set[str]:
        # ...
