from __future__ import annotations
import logging
import re
import urllib
from datetime import datetime
from re import Pattern
from typing import Any, TYPE_CHECKING, TypedDict
import pandas as pd
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from deprecation import deprecated
from flask_babel import gettext as __
from marshmallow import fields, Schema
from marshmallow.exceptions import ValidationError
from sqlalchemy import column, types
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import sqltypes
from superset.constants import TimeGrain
from superset.databases.schemas import encrypted_field_properties, EncryptedString
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.base import BaseEngineSpec, BasicPropertiesType
from superset.db_engine_specs.exceptions import SupersetDBAPIConnectionError
from superset.errors import SupersetError, SupersetErrorType
from superset.exceptions import SupersetException
from superset.sql.parse import SQLScript
from superset.sql_parse import Table
from superset.superset_typing import ResultSetColumnType
from superset.utils import core as utils, json
from superset.utils.hashing import md5_sha_from_str
try:
    import google.auth
    from google.cloud import bigquery
    from google.oauth2 import service_account
    dependencies_installed: bool = True
except ImportError:
    dependencies_installed: bool = False
try:
    import pandas_gbq
    can_upload: bool = True
except ModuleNotFoundError:
    can_upload: bool = False
if TYPE_CHECKING:
    from superset.models.core import Database
logger: logging.Logger = logging.getLogger()
CONNECTION_DATABASE_PERMISSIONS_REGEX: Pattern = re.compile('Access Denied: Project (?P<project_name>.+?): User does not have ' + 'bigquery.jobs.create permission in project (?P<project>.+?)')
TABLE_DOES_NOT_EXIST_REGEX: Pattern = re.compile('Table name "(?P<table>.*?)" missing dataset while no default dataset is set in the request')
COLUMN_DOES_NOT_EXIST_REGEX: Pattern = re.compile('Unrecognized name: (?P<column>.*?) at \\[(?P<location>.+?)\\]')
SCHEMA_DOES_NOT_EXIST_REGEX: Pattern = re.compile('bigquery error: 404 Not found: Dataset (?P<dataset>.*?):(?P<schema>.*?) was not found in location')
SYNTAX_ERROR_REGEX: Pattern = re.compile('Syntax error: Expected end of input but got identifier "(?P<syntax_error>.+?)"')
ma_plugin: MarshmallowPlugin = MarshmallowPlugin()

class BigQueryParametersSchema(Schema):
    credentials_info: EncryptedString = EncryptedString(required=False, metadata={'description': 'Contents of BigQuery JSON credentials.'})
    query: fields.Dict = fields.Dict(required=False)

class BigQueryParametersType(TypedDict):
    pass

class BigQueryEngineSpec(BaseEngineSpec):
    engine: str = 'bigquery'
    engine_name: str = 'Google BigQuery'
    max_column_name_length: int = 128
    disable_ssh_tunneling: bool = True
    parameters_schema: BigQueryParametersSchema = BigQueryParametersSchema()
    default_driver: str = 'bigquery'
    sqlalchemy_uri_placeholder: str = 'bigquery://{project_id}'
    run_multiple_statements_as_one: bool = True
    allows_hidden_cc_in_orderby: bool = True
    supports_catalog: bool = supports_dynamic_catalog: bool = True
    encrypted_extra_sensitive_fields: dict = {'$.credentials_info.private_key'}
    arraysize: int = 5000
    _date_trunc_functions: dict = {'DATE': 'DATE_TRUNC', 'DATETIME': 'DATETIME_TRUNC', 'TIME': 'TIME_TRUNC', 'TIMESTAMP': 'TIMESTAMP_TRUNC'}
    _time_grain_expressions: dict = {None: '{col}', TimeGrain.SECOND: 'CAST(TIMESTAMP_SECONDS(UNIX_SECONDS(CAST({col} AS TIMESTAMP))) AS {type})', TimeGrain.MINUTE: 'CAST(TIMESTAMP_SECONDS(60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 60)) AS {type})', TimeGrain.FIVE_MINUTES: 'CAST(TIMESTAMP_SECONDS(5*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 5*60)) AS {type})', TimeGrain.TEN_MINUTES: 'CAST(TIMESTAMP_SECONDS(10*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 10*60)) AS {type})', TimeGrain.FIFTEEN_MINUTES: 'CAST(TIMESTAMP_SECONDS(15*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 15*60)) AS {type})', TimeGrain.THIRTY_MINUTES: 'CAST(TIMESTAMP_SECONDS(30*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 30*60)) AS {type})', TimeGrain.HOUR: '{func}({col}, HOUR)', TimeGrain.DAY: '{func}({col}, DAY)', TimeGrain.WEEK: '{func}({col}, WEEK)', TimeGrain.WEEK_STARTING_MONDAY: '{func}({col}, ISOWEEK)', TimeGrain.MONTH: '{func}({col}, MONTH)', TimeGrain.QUARTER: '{func}({col}, QUARTER)', TimeGrain.YEAR: '{func}({col}, YEAR)'}
    custom_errors: dict = {CONNECTION_DATABASE_PERMISSIONS_REGEX: (__('Unable to connect. Verify that the following roles are set on the service account: "BigQuery Data Viewer", "BigQuery Metadata Viewer", "BigQuery Job User" and the following permissions are set "bigquery.readsessions.create", "bigquery.readsessions.getData"'), SupersetErrorType.CONNECTION_DATABASE_PERMISSIONS_ERROR, {}), TABLE_DOES_NOT_EXIST_REGEX: (__('The table "%(table)s" does not exist. A valid table must be used to run this query.'), SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR, {}), COLUMN_DOES_NOT_EXIST_REGEX: (__('We can\'t seem to resolve column "%(column)s" at line %(location)s.'), SupersetErrorType.COLUMN_DOES_NOT_EXIST_ERROR, {}), SCHEMA_DOES_NOT_EXIST_REGEX: (__('The schema "%(schema)s" does not exist. A valid schema must be used to run this query.'), SupersetErrorType.SCHEMA_DOES_NOT_EXIST_ERROR, {}), SYNTAX_ERROR_REGEX: (__('Please check your query for syntax errors at or near "%(syntax_error)s". Then, try running your query again.'), SupersetErrorType.SYNTAX_ERROR, {})}

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Any = None) -> str:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"CAST('{dttm.isoformat(timespec='microseconds')}' AS TIMESTAMP)"
        if isinstance(sqla_type, types.DateTime):
            return f"CAST('{dttm.isoformat(timespec='microseconds')}' AS DATETIME)"
        if isinstance(sqla_type, types.Time):
            return f"CAST('{dttm.strftime('%H:%M:%S.%f')}' AS TIME)"
        return None

    @classmethod
    def fetch_data(cls, cursor: Any, limit: int = None) -> Any:
        data = super().fetch_data(cursor, limit)
        if data and type(data[0]).__name__ == 'Row':
            data = [r.values() for r in data]
        return data

    @staticmethod
    def _mutate_label(label: str) -> str:
        ...

    @classmethod
    def _truncate_label(cls, label: str) -> str:
        ...

    @classmethod
    @deprecated(deprecated_in='3.0')
    def normalize_indexes(cls, indexes: Any) -> Any:
        ...

    @classmethod
    def get_indexes(cls, database: Database, inspector: Inspector, table: Table) -> Any:
        ...

    @classmethod
    def get_extra_table_metadata(cls, database: Database, table: Table) -> Any:
        ...

    @classmethod
    def epoch_to_dttm(cls) -> str:
        ...

    @classmethod
    def epoch_ms_to_dttm(cls) -> str:
        ...

    @classmethod
    def df_to_sql(cls, database: Database, table: Table, df: pd.DataFrame, to_sql_kwargs: Any) -> None:
        ...

    @classmethod
    def _get_client(cls, engine: Engine, database: Database) -> Any:
        ...

    @classmethod
    def estimate_query_cost(cls, database: Database, catalog: str, schema: str, sql: str, source: Any = None) -> Any:
        ...

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        ...

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> set:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Any, catalog: str = None, schema: str = None) -> tuple:
        ...

    @classmethod
    def get_allow_cost_estimate(cls, extra: Any) -> bool:
        ...

    @classmethod
    def custom_estimate_statement_cost(cls, statement: str, client: Any) -> dict:
        ...

    @classmethod
    def query_cost_formatter(cls, raw_cost: Any) -> Any:
        ...

    @classmethod
    def build_sqlalchemy_uri(cls, parameters: dict, encrypted_extra: Any = None) -> str:
        ...

    @classmethod
    def get_parameters_from_uri(cls, uri: str, encrypted_extra: Any = None) -> dict:
        ...

    @classmethod
    def get_dbapi_exception_mapping(cls) -> dict:
        ...

    @classmethod
    def validate_parameters(cls, properties: Any) -> list:
        ...

    @classmethod
    def parameters_json_schema(cls) -> dict:
        ...

    @classmethod
    def select_star(cls, database: Database, table: Table, engine: Engine, limit: int = 100, show_cols: bool = False, indent: bool = True, latest_partition: bool = True, cols: Any = None) -> Any:
        ...

    @classmethod
    def _get_fields(cls, cols: Any) -> Any:
        ...

    @classmethod
    def parse_error_exception(cls, exception: Exception) -> Exception:
        ...
