from __future__ import annotations
import logging
import re
import urllib
from datetime import datetime
from re import Pattern
from typing import Any, TYPE_CHECKING, Dict, List, Optional, Tuple, TypedDict
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

if TYPE_CHECKING:
    from superset.models.core import Database

logger: logging.Logger = logging.getLogger()
CONNECTION_DATABASE_PERMISSIONS_REGEX: Pattern = re.compile(
    'Access Denied: Project (?P<project_name>.+?): User does not have '
    'bigquery.jobs.create permission in project (?P<project>.+?)'
)
TABLE_DOES_NOT_EXIST_REGEX: Pattern = re.compile(
    'Table name "(?P<table>.*?)" missing dataset while no default dataset is set in the request'
)
COLUMN_DOES_NOT_EXIST_REGEX: Pattern = re.compile(
    'Unrecognized name: (?P<column>.*?) at \\[(?P<location>.+?)\\]'
)
SCHEMA_DOES_NOT_EXIST_REGEX: Pattern = re.compile(
    'bigquery error: 404 Not found: Dataset (?P<dataset>.*?):(?P<schema>.*?) was not found in location'
)
SYNTAX_ERROR_REGEX: Pattern = re.compile(
    'Syntax error: Expected end of input but got identifier "(?P<syntax_error>.+?)"'
)
ma_plugin: MarshmallowPlugin = MarshmallowPlugin()


class BigQueryParametersSchema(Schema):
    credentials_info: Optional[EncryptedString] = EncryptedString(
        required=False,
        metadata={'description': 'Contents of BigQuery JSON credentials.'}
    )
    query: Optional[Dict[str, Any]] = fields.Dict(required=False)


class BigQueryParametersType(TypedDict):
    credentials_info: Optional[str]
    query: Optional[Dict[str, Any]]


class BigQueryEngineSpec(BaseEngineSpec):
    """Engine spec for Google's BigQuery

    As contributed by @mxmzdlv on issue #945
    """
    engine: str = 'bigquery'
    engine_name: str = 'Google BigQuery'
    max_column_name_length: int = 128
    disable_ssh_tunneling: bool = True
    parameters_schema: BigQueryParametersSchema = BigQueryParametersSchema()
    default_driver: str = 'bigquery'
    sqlalchemy_uri_placeholder: str = 'bigquery://{project_id}'
    run_multiple_statements_as_one: bool = True
    allows_hidden_cc_in_orderby: bool = True
    supports_catalog: bool = True
    supports_dynamic_catalog: bool = True
    encrypted_extra_sensitive_fields: set = {'$.credentials_info.private_key'}
    """
    https://www.python.org/dev/peps/pep-0249/#arraysize
    raw_connections bypass the sqlalchemy-bigquery query execution context and deal with
    raw dbapi connection directly.
    If this value is not set, the default value is set to 1, as described here,
    https://googlecloudplatform.github.io/google-cloud-python/latest/_modules/google/cloud/bigquery/dbapi/cursor.html#Cursor

    The default value of 5000 is derived from the sqlalchemy-bigquery.
    https://github.com/googleapis/python-bigquery-sqlalchemy/blob/4e17259088f89eac155adc19e0985278a29ecf9c/sqlalchemy_bigquery/base.py#L762
    """
    arraysize: int = 5000
    _date_trunc_functions: Dict[str, str] = {
        'DATE': 'DATE_TRUNC', 
        'DATETIME': 'DATETIME_TRUNC', 
        'TIME': 'TIME_TRUNC', 
        'TIMESTAMP': 'TIMESTAMP_TRUNC'
    }
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = {
        None: '{col}',
        TimeGrain.SECOND: 'CAST(TIMESTAMP_SECONDS(UNIX_SECONDS(CAST({col} AS TIMESTAMP))) AS {type})',
        TimeGrain.MINUTE: 'CAST(TIMESTAMP_SECONDS(60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 60)) AS {type})',
        TimeGrain.FIVE_MINUTES: 'CAST(TIMESTAMP_SECONDS(5*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 5*60)) AS {type})',
        TimeGrain.TEN_MINUTES: 'CAST(TIMESTAMP_SECONDS(10*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 10*60)) AS {type})',
        TimeGrain.FIFTEEN_MINUTES: 'CAST(TIMESTAMP_SECONDS(15*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 15*60)) AS {type})',
        TimeGrain.THIRTY_MINUTES: 'CAST(TIMESTAMP_SECONDS(30*60 * DIV(UNIX_SECONDS(CAST({col} AS TIMESTAMP)), 30*60)) AS {type})',
        TimeGrain.HOUR: '{func}({col}, HOUR)',
        TimeGrain.DAY: '{func}({col}, DAY)',
        TimeGrain.WEEK: '{func}({col}, WEEK)',
        TimeGrain.WEEK_STARTING_MONDAY: '{func}({col}, ISOWEEK)',
        TimeGrain.MONTH: '{func}({col}, MONTH)',
        TimeGrain.QUARTER: '{func}({col}, QUARTER)',
        TimeGrain.YEAR: '{func}({col}, YEAR)'
    }
    custom_errors: Dict[Pattern, Tuple[str, SupersetErrorType, Dict[str, Any]]] = {
        CONNECTION_DATABASE_PERMISSIONS_REGEX: (
            __('Unable to connect. Verify that the following roles are set on the service account: "BigQuery Data Viewer", "BigQuery Metadata Viewer", "BigQuery Job User" and the following permissions are set "bigquery.readsessions.create", "bigquery.readsessions.getData"'), 
            SupersetErrorType.CONNECTION_DATABASE_PERMISSIONS_ERROR, 
            {}
        ),
        TABLE_DOES_NOT_EXIST_REGEX: (
            __('The table "%(table)s" does not exist. A valid table must be used to run this query.'),
            SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR, 
            {}
        ),
        COLUMN_DOES_NOT_EXIST_REGEX: (
            __('We can\'t seem to resolve column "%(column)s" at line %(location)s.'),
            SupersetErrorType.COLUMN_DOES_NOT_EXIST_ERROR, 
            {}
        ),
        SCHEMA_DOES_NOT_EXIST_REGEX: (
            __('The schema "%(schema)s" does not exist. A valid schema must be used to run this query.'),
            SupersetErrorType.SCHEMA_DOES_NOT_EXIST_ERROR, 
            {}
        ),
        SYNTAX_ERROR_REGEX: (
            __('Please check your query for syntax errors at or near "%(syntax_error)s". Then, try running your query again.'),
            SupersetErrorType.SYNTAX_ERROR, 
            {}
        )
    }

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[Any] = None) -> Optional[str]:
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
    def fetch_data(cls, cursor: Any, limit: Optional[int] = None) -> List[Any]:
        data = super().fetch_data(cursor, limit)
        if data and type(data[0]).__name__ == 'Row':
            data = [tuple(r.values()) for r in data]
        return data

    @staticmethod
    def _mutate_label(label: str) -> str:
        """
        BigQuery field_name should start with a letter or underscore and contain only
        alphanumeric characters. Labels that start with a number are prefixed with an
        underscore. Any unsupported characters are replaced with underscores and an
        md5 hash is added to the end of the label to avoid possible collisions.

        :param label: Expected expression label
        :return: Conditionally mutated label
        """
        label_hashed: str = '_' + md5_sha_from_str(label)
        label_mutated: str = '_' + label if re.match(r'^\d', label) else label
        label_mutated = re.sub(r'[^\w]+', '_', label_mutated)
        if label_mutated != label:
            label_mutated += label_hashed[:6]
        return label_mutated

    @classmethod
    def _truncate_label(cls, label: str) -> str:
        """BigQuery requires column names start with either a letter or
        underscore. To make sure this is always the case, an underscore is prefixed
        to the md5 hash of the original label.

        :param label: expected expression label
        :return: truncated label
        """
        return '_' + md5_sha_from_str(label)

    @classmethod
    @deprecated(deprecated_in='3.0')
    def normalize_indexes(cls, indexes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalizes indexes for more consistency across db engines

        :param indexes: Raw indexes as returned by SQLAlchemy
        :return: cleaner, more aligned index definition
        """
        normalized_idxs: List[Dict[str, Any]] = []
        for ix in indexes:
            column_names: List[Optional[str]] = ix.get('column_names') or []
            ix['column_names'] = [col for col in column_names if col is not None]
            if ix['column_names']:
                normalized_idxs.append(ix)
        return normalized_idxs

    @classmethod
    def get_indexes(cls, database: Database, inspector: Inspector, table: Table) -> List[Dict[str, Any]]:
        """
        Get the indexes associated with the specified schema/table.

        :param database: The database to inspect
        :param inspector: The SQLAlchemy inspector
        :param table: The table instance to inspect
        :returns: The indexes
        """
        return cls.normalize_indexes(inspector.get_indexes(table.table, table.schema))

    @classmethod
    def get_extra_table_metadata(cls, database: Database, table: Table) -> Dict[str, Any]:
        indexes: List[Dict[str, Any]] = database.get_indexes(table)
        if not indexes:
            return {}
        partitions_columns: List[List[str]] = [
            index.get('column_names', []) for index in indexes if index.get('name') == 'partition'
        ]
        cluster_columns: List[List[str]] = [
            index.get('column_names', []) for index in indexes if index.get('name') == 'clustering'
        ]
        return {'partitions': {'cols': partitions_columns}, 'clustering': {'cols': cluster_columns}}

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return 'TIMESTAMP_SECONDS({col})'

    @classmethod
    def epoch_ms_to_dttm(cls) -> str:
        return 'TIMESTAMP_MILLIS({col})'

    @classmethod
    def df_to_sql(
        cls, 
        database: Database, 
        table: Table, 
        df: pd.DataFrame, 
        to_sql_kwargs: Dict[str, Any]
    ) -> None:
        """
            Upload data from a Pandas DataFrame to a database.

            Calls `pandas_gbq.DataFrame.to_gbq` which requires `pandas_gbq` to be installed.

            Note this method does not create metadata for the table.

            :param database: The database to upload the data to
            :param table: The table to upload the data to
            :param df: The dataframe with data to be uploaded
            :param to_sql_kwargs: The kwargs to be passed to pandas.DataFrame.to_sql` method
        """
        if not can_upload:
            raise SupersetException('Could not import libraries needed to upload data to BigQuery.')
        if not table.schema:
            raise SupersetException('The table schema must be defined')
        to_gbq_kwargs: Dict[str, Any] = {}
        with cls.get_engine(database, catalog=table.catalog, schema=table.schema) as engine:
            to_gbq_kwargs = {
                'destination_table': str(table), 
                'project_id': engine.url.host
            }
        if (creds := engine.dialect.credentials_info):
            to_gbq_kwargs['credentials'] = service_account.Credentials.from_service_account_info(creds)
        supported_kwarg_keys: set = {'if_exists'}
        for key in supported_kwarg_keys:
            if key in to_sql_kwargs:
                to_gbq_kwargs[key] = to_sql_kwargs[key]
        pandas_gbq.to_gbq(df, **to_gbq_kwargs)

    @classmethod
    def _get_client(cls, engine: Engine, database: Database) -> bigquery.Client:
        """
        Return the BigQuery client associated with an engine.
        """
        if not dependencies_installed:
            raise SupersetException('Could not import libraries needed to connect to BigQuery.')
        if (credentials_info := engine.dialect.credentials_info):
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            return bigquery.Client(credentials=credentials)
        try:
            credentials = google.auth.default()[0]
            return bigquery.Client(credentials=credentials)
        except google.auth.exceptions.DefaultCredentialsError as ex:
            raise SupersetDBAPIConnectionError('The database credentials could not be found.') from ex

    @classmethod
    def estimate_query_cost(
        cls, 
        database: Database, 
        catalog: str, 
        schema: str, 
        sql: str, 
        source: Optional[str] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Estimate the cost of a multiple statement SQL query.

        :param database: Database instance
        :param catalog: Database project
        :param schema: Database schema
        :param sql: SQL query with possibly multiple statements
        :param source: Source of the query (eg, "sql_lab")
        """
        extra: Dict[str, Any] = database.get_extra() or {}
        if not cls.get_allow_cost_estimate(extra):
            raise SupersetException('Database does not support cost estimation')
        parsed_script: SQLScript = SQLScript(sql, engine=cls.engine)
        with cls.get_engine(database, catalog=catalog, schema=schema) as engine:
            client: bigquery.Client = cls._get_client(engine, database)
            return [
                cls.custom_estimate_statement_cost(
                    cls.process_statement(statement, database), 
                    client
                ) for statement in parsed_script.statements
            ]

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        """
        Get the default catalog.
        """
        url: URL = database.url_object
        if (project := (url.host or url.database)):
            return project
        with database.get_sqla_engine() as engine:
            client: bigquery.Client = cls._get_client(engine, database)
            return client.project

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> set:
        """
        Get all catalogs.

        In BigQuery, a catalog is called a "project".
        """
        with database.get_sqla_engine() as engine:
            try:
                client: bigquery.Client = cls._get_client(engine, database)
            except SupersetDBAPIConnectionError:
                logger.warning(
                    'Could not connect to database to get catalogs due to missing credentials. This is normal in certain circumstances, for example, doing an import.'
                )
                return set()
            projects: List[bigquery.Project] = list(client.list_projects())
        return {project.project_id for project in projects}

    @classmethod
    def adjust_engine_params(
        cls, 
        uri: URL, 
        connect_args: Dict[str, Any], 
        catalog: Optional[str] = None, 
        schema: Optional[str] = None
    ) -> Tuple[URL, Dict[str, Any]]:
        if catalog:
            uri = uri.set(host=catalog, database='')
        return uri, connect_args

    @classmethod
    def get_allow_cost_estimate(cls, extra: Dict[str, Any]) -> bool:
        return True

    @classmethod
    def custom_estimate_statement_cost(
        cls, 
        statement: str, 
        client: bigquery.Client
    ) -> Dict[str, Union[str, float]]:
        """
        Custom version that receives a client instead of a cursor.
        """
        job_config: bigquery.QueryJobConfig = bigquery.QueryJobConfig(dry_run=True)
        query_job: bigquery.QueryJob = client.query(statement, job_config=job_config)
        byte_division: int = 1024
        if hasattr(query_job, 'total_bytes_processed'):
            query_bytes_processed: int = query_job.total_bytes_processed
            if query_bytes_processed // byte_division == 0:
                byte_type: str = 'B'
                total_bytes_processed: int = query_bytes_processed
            elif query_bytes_processed // byte_division ** 2 == 0:
                byte_type = 'KB'
                total_bytes_processed = round(query_bytes_processed / byte_division, 2)
            elif query_bytes_processed // byte_division ** 3 == 0:
                byte_type = 'MB'
                total_bytes_processed = round(query_bytes_processed / byte_division ** 2, 2)
            else:
                byte_type = 'GB'
                total_bytes_processed = round(query_bytes_processed / byte_division ** 3, 2)
            return {f'{byte_type} Processed': total_bytes_processed}
        return {}

    @classmethod
    def query_cost_formatter(cls, raw_cost: List[Dict[str, Union[str, float]]]) -> List[Dict[str, str]]:
        return [{k: str(v) for k, v in row.items()} for row in raw_cost]

    @classmethod
    def build_sqlalchemy_uri(
        cls, 
        parameters: Dict[str, Any], 
        encrypted_extra: Optional[Dict[str, Any]] = None
    ) -> str:
        query: Dict[str, Any] = parameters.get('query', {})
        query_params: str = urllib.parse.urlencode(query)
        if encrypted_extra:
            credentials_info: Optional[Dict[str, Any]] = encrypted_extra.get('credentials_info')
            if isinstance(credentials_info, str):
                credentials_info = json.loads(credentials_info)
            project_id: Optional[str] = credentials_info.get('project_id') if credentials_info else None
        if not encrypted_extra:
            raise ValidationError('Missing service credentials')
        if project_id:
            return f'{cls.default_driver}://{project_id}/?{query_params}'
        raise ValidationError('Invalid service credentials')

    @classmethod
    def get_parameters_from_uri(
        cls, 
        uri: URL, 
        encrypted_extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        value: URL = make_url_safe(uri)
        if encrypted_extra:
            return {**encrypted_extra, 'query': dict(value.query)}
        raise ValidationError('Invalid service credentials')

    @classmethod
    def get_dbapi_exception_mapping(cls) -> Dict[Type[BaseException], Type[SupersetDBAPIConnectionError]]:
        from google.auth.exceptions import DefaultCredentialsError
        return {DefaultCredentialsError: SupersetDBAPIConnectionError}

    @classmethod
    def validate_parameters(cls, properties: Dict[str, Any]) -> List[str]:
        return []

    @classmethod
    def parameters_json_schema(cls) -> Optional[Dict[str, Any]]:
        """
        Return configuration parameters as OpenAPI.
        """
        if not cls.parameters_schema:
            return None
        spec = APISpec(
            title='Database Parameters', 
            version='1.0.0', 
            openapi_version='3.0.0', 
            plugins=[ma_plugin]
        )
        ma_plugin.init_spec(spec)
        ma_plugin.converter.add_attribute_function(encrypted_field_properties)
        spec.components.schema(cls.__name__, schema=cls.parameters_schema)
        return spec.to_dict()['components']['schemas'][cls.__name__]

    @classmethod
    def select_star(
        cls, 
        database: Database, 
        table: Table, 
        engine: Engine, 
        limit: int = 100, 
        show_cols: bool = False, 
        indent: bool = True, 
        latest_partition: bool = True, 
        cols: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Remove array structures from `SELECT *`.

        BigQuery supports structures and arrays of structures, eg:

            author STRUCT<name STRING, email STRING>
            trailer ARRAY<STRUCT<key STRING, value STRING>>

        When loading metadata for a table each key in the struct is displayed as a
        separate pseudo-column, eg:

            - author
            - author.name
            - author.email
            - trailer
            - trailer.key
            - trailer.value

        When generating the `SELECT *` statement we want to remove any keys from
        structs inside an array, since selecting them results in an error. The correct
        select statement should look like this:

            SELECT
              `author`,
              `author`.`name`,
              `author`.`email`,
              `trailer`
            FROM
              table

        Selecting `trailer.key` or `trailer.value` results in an error, as opposed to
        selecting `author.name`, since they are keys in a structure inside an array.

        This method removes any array pseudo-columns.
        """
        if cols:
            array_prefixes: set = {col['column_name'] for col in cols if isinstance(col['type'], sqltypes.ARRAY)}
            cols = [
                col for col in cols 
                if '.' not in col['column_name'] or col['column_name'].split('.')[0] not in array_prefixes
            ]
        return super().select_star(database, table, engine, limit, show_cols, indent, latest_partition, cols)

    @classmethod
    def _get_fields(cls, cols: List[Dict[str, Any]]) -> List[Any]:
        """
        Label columns using their fully qualified name.

        BigQuery supports columns of type `struct`, which are basically dictionaries.
        When loading metadata for a table with struct columns, each key in the struct
        is displayed as a separate pseudo-column, eg:

            author STRUCT<name STRING, email STRING>

        Will be shown as 3 columns:

            - author
            - author.name
            - author.email

        If we select those fields:

            SELECT `author`, `author`.`name`, `author`.`email` FROM table

        The resulting columns will be called "author", "name", and "email", This may
        result in a clash with other columns. To prevent that, we explicitly label
        the columns using their fully qualified name, so we end up with "author",
        "author__name" and "author__email", respectively.
        """
        return [
            column(c['column_name']).label(c['column_name'].replace('.', '__')) 
            for c in cols
        ]

    @classmethod
    def parse_error_exception(cls, exception: Exception) -> Exception:
        try:
            return type(exception)(str(exception).splitlines()[0].strip())
        except Exception:
            return exception
