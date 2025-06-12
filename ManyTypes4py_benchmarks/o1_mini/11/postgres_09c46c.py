from __future__ import annotations
import logging
import re
from datetime import datetime
from re import Pattern
from typing import Any, TYPE_CHECKING, Dict, Optional, Tuple, List, Set
from flask_babel import gettext as __
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, ENUM, JSON
from sqlalchemy.dialects.postgresql.base import PGInspector
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.types import Date, DateTime, String
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec, BasicParametersMixin
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import SupersetException, SupersetSecurityException
from superset.models.sql_lab import Query
from superset.sql.parse import SQLScript
from superset.utils import core as utils, json
from superset.utils.core import GenericDataType

if TYPE_CHECKING:
    from superset.models.core import Database

logger = logging.getLogger()
CONNECTION_INVALID_USERNAME_REGEX: Pattern = re.compile('role "(?P<username>.*?)" does not exist')
CONNECTION_INVALID_PASSWORD_REGEX: Pattern = re.compile('password authentication failed for user "(?P<username>.*?)"')
CONNECTION_INVALID_PASSWORD_NEEDED_REGEX: Pattern = re.compile('no password supplied')
CONNECTION_INVALID_HOSTNAME_REGEX: Pattern = re.compile('could not translate host name "(?P<hostname>.*?)" to address: nodename nor servname provided, or not known')
CONNECTION_PORT_CLOSED_REGEX: Pattern = re.compile('could not connect to server: Connection refused\\s+Is the server running on host "(?P<hostname>.*?)" (\\(.*?\\) )?and accepting\\s+TCP/IP connections on port (?P<port>.*?)\\?')
CONNECTION_HOST_DOWN_REGEX: Pattern = re.compile('could not connect to server: (?P<reason>.*?)\\s+Is the server running on host "(?P<hostname>.*?)" (\\(.*?\\) )?and accepting\\s+TCP/IP connections on port (?P<port>.*?)\\?')
CONNECTION_UNKNOWN_DATABASE_REGEX: Pattern = re.compile('database "(?P<database>.*?)" does not exist')
COLUMN_DOES_NOT_EXIST_REGEX: Pattern = re.compile('postgresql error: column "(?P<column_name>.+?)" does not exist\\s+LINE (?P<location>\\d+?)')
SYNTAX_ERROR_REGEX: Pattern = re.compile('syntax error at or near "(?P<syntax_error>.*?)"')


def parse_options(connect_args: Dict[str, Any]) -> Dict[str, str]:
    """
    Parse ``options`` from  ``connect_args`` into a dictionary.
    """
    if not isinstance(connect_args.get('options'), str):
        return {}
    tokens = (
        tuple((token.strip() for token in option.strip().split('=', 1)))
        for option in re.split('-c\\s?', connect_args['options'])
        if '=' in option
    )
    return {token[0]: token[1] for token in tokens}


class PostgresBaseEngineSpec(BaseEngineSpec):
    """Abstract class for Postgres 'like' databases"""
    engine: str = ''
    engine_name: str = 'PostgreSQL'
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = {
        None: '{col}',
        TimeGrain.SECOND: "DATE_TRUNC('second', {col})",
        TimeGrain.FIVE_SECONDS: "DATE_TRUNC('minute', {col}) + INTERVAL '5 seconds' * FLOOR(EXTRACT(SECOND FROM {col}) / 5)",
        TimeGrain.THIRTY_SECONDS: "DATE_TRUNC('minute', {col}) + INTERVAL '30 seconds' * FLOOR(EXTRACT(SECOND FROM {col}) / 30)",
        TimeGrain.MINUTE: "DATE_TRUNC('minute', {col})",
        TimeGrain.FIVE_MINUTES: "DATE_TRUNC('hour', {col}) + INTERVAL '5 minutes' * FLOOR(EXTRACT(MINUTE FROM {col}) / 5)",
        TimeGrain.TEN_MINUTES: "DATE_TRUNC('hour', {col}) + INTERVAL '10 minutes' * FLOOR(EXTRACT(MINUTE FROM {col}) / 10)",
        TimeGrain.FIFTEEN_MINUTES: "DATE_TRUNC('hour', {col}) + INTERVAL '15 minutes' * FLOOR(EXTRACT(MINUTE FROM {col}) / 15)",
        TimeGrain.THIRTY_MINUTES: "DATE_TRUNC('hour', {col}) + INTERVAL '30 minutes' * FLOOR(EXTRACT(MINUTE FROM {col}) / 30)",
        TimeGrain.HOUR: "DATE_TRUNC('hour', {col})",
        TimeGrain.DAY: "DATE_TRUNC('day', {col})",
        TimeGrain.WEEK: "DATE_TRUNC('week', {col})",
        TimeGrain.MONTH: "DATE_TRUNC('month', {col})",
        TimeGrain.QUARTER: "DATE_TRUNC('quarter', {col})",
        TimeGrain.YEAR: "DATE_TRUNC('year', {col})",
    }
    custom_errors: Dict[Pattern, Tuple[str, SupersetErrorType, Dict[str, List[str]]]] = {
        CONNECTION_INVALID_USERNAME_REGEX: (
            __('The username "%(username)s" does not exist.'),
            SupersetErrorType.CONNECTION_INVALID_USERNAME_ERROR,
            {'invalid': ['username']}
        ),
        CONNECTION_INVALID_PASSWORD_REGEX: (
            __('The password provided for username "%(username)s" is incorrect.'),
            SupersetErrorType.CONNECTION_INVALID_PASSWORD_ERROR,
            {'invalid': ['username', 'password']}
        ),
        CONNECTION_INVALID_PASSWORD_NEEDED_REGEX: (
            __('Please re-enter the password.'),
            SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR,
            {'invalid': ['password']}
        ),
        CONNECTION_INVALID_HOSTNAME_REGEX: (
            __('The hostname "%(hostname)s" cannot be resolved.'),
            SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR,
            {'invalid': ['host']}
        ),
        CONNECTION_PORT_CLOSED_REGEX: (
            __('Port %(port)s on hostname "%(hostname)s" refused the connection.'),
            SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR,
            {'invalid': ['host', 'port']}
        ),
        CONNECTION_HOST_DOWN_REGEX: (
            __('The host "%(hostname)s" might be down, and can\'t be reached on port %(port)s.'),
            SupersetErrorType.CONNECTION_HOST_DOWN_ERROR,
            {'invalid': ['host', 'port']}
        ),
        CONNECTION_UNKNOWN_DATABASE_REGEX: (
            __('Unable to connect to database "%(database)s".'),
            SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR,
            {'invalid': ['database']}
        ),
        COLUMN_DOES_NOT_EXIST_REGEX: (
            __('We can\'t seem to resolve the column "%(column_name)s" at line %(location)s.'),
            SupersetErrorType.COLUMN_DOES_NOT_EXIST_ERROR,
            {}
        ),
        SYNTAX_ERROR_REGEX: (
            __('Please check your query for syntax errors at or near "%(syntax_error)s". Then, try running your query again.'),
            SupersetErrorType.SYNTAX_ERROR,
            {}
        ),
    }

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int] = None) -> List[Any]:
        if not cursor.description:
            return []
        return super().fetch_data(cursor, limit)

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return "(timestamp 'epoch' + {col} * interval '1 second')"

    @classmethod
    def convert_dttm(cls, target_type: Any, dttm: datetime, db_extra: Optional[Any] = None) -> Optional[str]:
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, Date):
            return f"TO_DATE('{dttm.date().isoformat()}', 'YYYY-MM-DD')"
        if isinstance(sqla_type, DateTime):
            dttm_formatted = dttm.isoformat(sep=' ', timespec='microseconds')
            return f"TO_TIMESTAMP('{dttm_formatted}', 'YYYY-MM-DD HH24:MI:SS.US')"
        return None


class PostgresEngineSpec(BasicParametersMixin, PostgresBaseEngineSpec):
    engine: str = 'postgresql'
    engine_aliases: Set[str] = {'postgres'}
    supports_dynamic_schema: bool = True
    supports_catalog: bool = True
    supports_dynamic_catalog: bool = True
    default_driver: str = 'psycopg2'
    sqlalchemy_uri_placeholder: str = 'postgresql://user:password@host:port/dbname[?key=value&key=value...]'
    encryption_parameters: Dict[str, str] = {'sslmode': 'require'}
    max_column_name_length: int = 63
    try_remove_schema_from_table_name: bool = False
    column_type_mappings: List[Tuple[Pattern, Any, GenericDataType]] = [
        (re.compile('^double precision', re.IGNORECASE), DOUBLE_PRECISION(), GenericDataType.NUMERIC),
        (re.compile('^array.*', re.IGNORECASE), String(), GenericDataType.STRING),
        (re.compile('^json.*', re.IGNORECASE), JSON(), GenericDataType.STRING),
        (re.compile('^enum.*', re.IGNORECASE), ENUM(), GenericDataType.STRING),
    ]

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> Optional[str]:
        """
        Return the configured schema.

        While Postgres doesn't support connecting directly to a given schema, it allows
        users to specify a "search path" that is used to resolve non-qualified table
        names; this can be specified in the database ``connect_args``.

        One important detail is that the search path can be a comma separated list of
        schemas. While this is supported by the SQLAlchemy dialect, it shouldn't be used
        in Superset because it breaks schema-level permissions, since it's impossible
        to determine the schema for a non-qualified table in a query. In cases like
        that we raise an exception.

        Note that because the DB engine supports dynamic schema this method is never
        called. It's left here as an implementation reference.
        """
        options = parse_options(connect_args)
        if (search_path := options.get('search_path')):
            schemas = search_path.split(',')
            if len(schemas) > 1:
                raise Exception('Multiple schemas are configured in the search path, which means Superset is unable to determine the schema of unqualified table names and enforce permissions.')
            return schemas[0]
        return None

    @classmethod
    def get_default_schema_for_query(cls, database: Database, query: Query) -> Optional[str]:
        """
        Return the default schema for a given query.

        This method simply uses the parent method after checking that there are no
        malicious path setting in the query.
        """
        script = SQLScript(query.sql, engine=cls.engine)
        settings = script.get_settings()
        if 'search_path' in settings:
            raise SupersetSecurityException(
                SupersetError(
                    error_type=SupersetErrorType.QUERY_SECURITY_ACCESS_ERROR,
                    message=__('Users are not allowed to set a search path for security reasons.'),
                    level=ErrorLevel.ERROR
                )
            )
        return super().get_default_schema_for_query(database, query)

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = None, schema: Optional[str] = None) -> Tuple[URL, Dict[str, Any]]:
        """
        Set the catalog (database).
        """
        if catalog:
            uri = uri.set(database=catalog)
        return (uri, connect_args)

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        """
        Return the default catalog for a given database.
        """
        return database.url_object.database

    @classmethod
    def get_prequeries(cls, database: Database, catalog: Optional[str] = None, schema: Optional[str] = None) -> List[str]:
        """
        Set the search path to the specified schema.

        This is important for two reasons: in SQL Lab it will allow queries to run in
        the schema selected in the dropdown, resolving unqualified table names to the
        expected schema.

        But more importantly, in SQL Lab this is used to check if the user has access to
        any tables with unqualified names. If the schema is not set by SQL Lab it could
        be anything, and we would have to block users from running any queries
        referencing tables without an explicit schema.
        """
        return [f'set search_path = "{schema}"'] if schema else []

    @classmethod
    def get_allow_cost_estimate(cls, extra: Any) -> bool:
        return True

    @classmethod
    def estimate_statement_cost(cls, database: Database, statement: str, cursor: Any) -> Dict[str, float]:
        """
        Run a SQL query that estimates the cost of a given statement.
        :param database: A Database object
        :param statement: A single SQL statement
        :param cursor: Cursor instance
        :return: JSON response from Trino
        """
        sql = f'EXPLAIN {statement}'
        cursor.execute(sql)
        result = cursor.fetchone()[0]
        match = re.search('cost=([\\d\\.]+)\\.\\.([\\d\\.]+)', result)
        if match:
            return {'Start-up cost': float(match.group(1)), 'Total cost': float(match.group(2))}
        return {}

    @classmethod
    def query_cost_formatter(cls, raw_cost: Dict[str, float]) -> List[Dict[str, str]]:
        return [{k: str(v) for k, v in row.items()} for row in [raw_cost]]

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> Set[str]:
        """
        Return all catalogs.

        In Postgres, a catalog is called a "database".
        """
        return {
            catalog for catalog, in inspector.bind.execute(
                '\nSELECT datname FROM pg_database\nWHERE datistemplate = false;\n            '
            )
        }

    @classmethod
    def get_table_names(cls, database: Database, inspector: Inspector, schema: str) -> Set[str]:
        """Need to consider foreign tables for PostgreSQL"""
        return set(inspector.get_table_names(schema)) | set(inspector.get_foreign_table_names(schema))

    @staticmethod
    def get_extra_params(database: Database) -> Dict[str, Any]:
        """
        For Postgres, the path to a SSL certificate is placed in `connect_args`.

        :param database: database instance from which to extract extras
        :raises CertificateException: If certificate is not valid/unparseable
        :raises SupersetException: If database extra json payload is unparseable
        """
        try:
            extra = json.loads(database.extra or '{}')
        except json.JSONDecodeError as ex:
            raise SupersetException('Unable to parse database extras') from ex
        if database.server_cert:
            engine_params = extra.get('engine_params', {})
            connect_args = engine_params.get('connect_args', {})
            connect_args['sslmode'] = connect_args.get('sslmode', 'verify-full')
            path = utils.create_ssl_cert_file(database.server_cert)
            connect_args['sslrootcert'] = path
            engine_params['connect_args'] = connect_args
            extra['engine_params'] = engine_params
        return extra

    @classmethod
    def get_datatype(cls, type_code: Any) -> Optional[str]:
        from psycopg2.extensions import binary_types, string_types
        types = binary_types.copy()
        types.update(string_types)
        if type_code in types:
            return types[type_code].name
        return None

    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Query) -> int:
        """
        Get Postgres PID that will be used to cancel all other running
        queries in the same session.

        :param cursor: Cursor instance in which the query will be executed
        :param query: Query instance
        :return: Postgres PID
        """
        cursor.execute('SELECT pg_backend_pid()')
        row = cursor.fetchone()
        return row[0] if row else -1

    @classmethod
    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: int) -> bool:
        """
        Cancel query in the underlying database.

        :param cursor: New cursor instance to the db of the query
        :param query: Query instance
        :param cancel_query_id: Postgres PID
        :return: True if query cancelled successfully, False otherwise
        """
        try:
            cursor.execute(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid='{cancel_query_id}'")
        except Exception:
            return False
        return True
