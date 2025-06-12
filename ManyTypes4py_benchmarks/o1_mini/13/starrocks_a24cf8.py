import logging
import re
from re import Pattern
from typing import Any, Optional, Tuple, Dict, List
from urllib import parse
from flask_babel import gettext as __
from sqlalchemy import Float, Integer, Numeric, types
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.type_api import TypeEngine
from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.errors import SupersetErrorType
from superset.models.core import Database
from superset.utils.core import GenericDataType

CONNECTION_ACCESS_DENIED_REGEX: Pattern = re.compile("Access denied for user '(?P<username>.*?)'")
CONNECTION_UNKNOWN_DATABASE_REGEX: Pattern = re.compile("Unknown database '(?P<database>.*?)'")
logger: logging.Logger = logging.getLogger(__name__)

class TINYINT(Integer):
    __visit_name__: str = 'TINYINT'

class LARGEINT(Integer):
    __visit_name__: str = 'LARGEINT'

class DOUBLE(Float):
    __visit_name__: str = 'DOUBLE'

class HLL(Numeric):
    __visit_name__: str = 'HLL'

class BITMAP(Numeric):
    __visit_name__: str = 'BITMAP'

class PERCENTILE(Numeric):
    __visit_name__: str = 'PERCENTILE'

class ARRAY(TypeEngine):
    __visit_name__: str = 'ARRAY'

    @property
    def python_type(self) -> type:
        return list

class MAP(TypeEngine):
    __visit_name__: str = 'MAP'

    @property
    def python_type(self) -> type:
        return dict

class STRUCT(TypeEngine):
    __visit_name__: str = 'STRUCT'

    @property
    def python_type(self) -> Optional[type]:
        return None

class StarRocksEngineSpec(MySQLEngineSpec):
    engine: str = 'starrocks'
    engine_name: str = 'StarRocks'
    default_driver: str = 'starrocks'
    sqlalchemy_uri_placeholder: str = 'starrocks://user:password@host:port/catalog.db[?key=value&key=value...]'
    column_type_mappings: List[Tuple[Pattern, TypeEngine, GenericDataType]] = [
        (re.compile('^tinyint', re.IGNORECASE), TINYINT(), GenericDataType.NUMERIC),
        (re.compile('^largeint', re.IGNORECASE), LARGEINT(), GenericDataType.NUMERIC),
        (re.compile('^decimal.*', re.IGNORECASE), types.DECIMAL(), GenericDataType.NUMERIC),
        (re.compile('^double', re.IGNORECASE), DOUBLE(), GenericDataType.NUMERIC),
        (re.compile('^varchar(\\((\\d+)\\))*$', re.IGNORECASE), types.VARCHAR(), GenericDataType.STRING),
        (re.compile('^char(\\((\\d+)\\))*$', re.IGNORECASE), types.CHAR(), GenericDataType.STRING),
        (re.compile('^json', re.IGNORECASE), types.JSON(), GenericDataType.STRING),
        (re.compile('^binary.*', re.IGNORECASE), types.String(), GenericDataType.STRING),
        (re.compile('^percentile', re.IGNORECASE), PERCENTILE(), GenericDataType.STRING),
        (re.compile('^hll', re.IGNORECASE), HLL(), GenericDataType.STRING),
        (re.compile('^bitmap', re.IGNORECASE), BITMAP(), GenericDataType.STRING),
        (re.compile('^array.*', re.IGNORECASE), ARRAY(), GenericDataType.STRING),
        (re.compile('^map.*', re.IGNORECASE), MAP(), GenericDataType.STRING),
        (re.compile('^struct.*', re.IGNORECASE), STRUCT(), GenericDataType.STRING),
    ]
    custom_errors: Dict[Pattern, Tuple[str, SupersetErrorType, Dict[str, List[str]]]] = {
        CONNECTION_ACCESS_DENIED_REGEX: (
            __('Either the username "%(username)s" or the password is incorrect.'),
            SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR,
            {'invalid': ['username', 'password']},
        ),
        CONNECTION_UNKNOWN_DATABASE_REGEX: (
            __('Unable to connect to database "%(database)s".'),
            SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR,
            {'invalid': ['database']},
        ),
    }

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None
    ) -> Tuple[URL, Dict[str, Any]]:
        database: str = uri.database
        if schema and database:
            schema = parse.quote(schema, safe='')
            if '.' in database:
                database = database.split('.')[0] + '.' + schema
            else:
                database = 'default_catalog.' + schema
            uri = uri.set(database=database)
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: Dict[str, Any]
    ) -> Optional[str]:
        """
        Return the configured schema.

        For StarRocks the SQLAlchemy URI looks like this:

            starrocks://localhost:9030/catalog.schema

        """
        database: str = sqlalchemy_uri.database.strip('/')
        if '.' not in database:
            return None
        return parse.unquote(database.split('.')[1])

    @classmethod
    def get_url_for_impersonation(
        cls,
        url: URL,
        impersonate_user: bool,
        username: Optional[str] = None,
        access_token: Optional[str] = None
    ) -> URL:
        """
        Return a modified URL with the username set.

        :param url: SQLAlchemy URL object
        :param impersonate_user: Flag indicating if impersonation is enabled
        :param username: Effective username
        :param access_token: Personal access token
        """
        return url

    @classmethod
    def get_prequeries(
        cls,
        database: Database,
        catalog: Optional[str] = None,
        schema: Optional[str] = None
    ) -> List[str]:
        """
        Return pre-session queries.

        These are currently used as an alternative to ``adjust_engine_params`` for
        databases where the selected schema cannot be specified in the SQLAlchemy URI or
        connection arguments.

        For example, in order to specify a default schema in RDS we need to run a query
        at the beginning of the session:

            sql> set search_path = my_schema;

        """
        if database.impersonate_user:
            username: Optional[str] = database.get_effective_user(database.url_object)
            if username:
                return [f'EXECUTE AS "{username}" WITH NO REVERT;']
        return []
