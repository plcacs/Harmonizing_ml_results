import logging
import re
from re import Pattern
from typing import Any, Optional, Dict, Tuple, Set, List
from typing import Pattern as TPattern
from urllib import parse
from flask_babel import gettext as __
from sqlalchemy import Float, Integer, Numeric, String, TEXT, types
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.type_api import TypeEngine
from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.errors import SupersetErrorType
from superset.models.core import Database
from superset.utils.core import GenericDataType

CONNECTION_ACCESS_DENIED_REGEX: TPattern[str] = re.compile("Access denied for user '(?P<username>.*?)'")
CONNECTION_INVALID_HOSTNAME_REGEX: TPattern[str] = re.compile("Unknown Doris server host '(?P<hostname>.*?)'")
CONNECTION_UNKNOWN_DATABASE_REGEX: TPattern[str] = re.compile("Unknown database '(?P<database>.*?)'")
CONNECTION_HOST_DOWN_REGEX: TPattern[str] = re.compile("Can't connect to Doris server on '(?P<hostname>.*?)'")
SYNTAX_ERROR_REGEX: TPattern[str] = re.compile("check the manual that corresponds to your MySQL server version for the right syntax to use near '(?P<server_error>.*)")
logger: logging.Logger = logging.getLogger(__name__)


class TINYINT(Integer):
    __visit_name__ = 'TINYINT'


class LARGEINT(Integer):
    __visit_name__ = 'LARGEINT'


class DOUBLE(Float):
    __visit_name__ = 'DOUBLE'


class HLL(Numeric):
    __visit_name__ = 'HLL'


class BITMAP(Numeric):
    __visit_name__ = 'BITMAP'


class QuantileState(Numeric):
    __visit_name__ = 'QUANTILE_STATE'


class AggState(Numeric):
    __visit_name__ = 'AGG_STATE'


class ARRAY(TypeEngine):
    __visit_name__ = 'ARRAY'

    @property
    def python_type(self) -> type:
        return list


class MAP(TypeEngine):
    __visit_name__ = 'MAP'

    @property
    def python_type(self) -> type:
        return dict


class STRUCT(TypeEngine):
    __visit_name__ = 'STRUCT'

    @property
    def python_type(self) -> Optional[type]:
        return None


class DorisEngineSpec(MySQLEngineSpec):
    engine: str = 'pydoris'
    engine_aliases: Set[str] = {'doris'}
    engine_name: str = 'Apache Doris'
    max_column_name_length: int = 64
    default_driver: str = 'pydoris'
    sqlalchemy_uri_placeholder: str = 'doris://user:password@host:port/catalog.db[?key=value&key=value...]'
    encryption_parameters: Dict[str, str] = {'ssl': '0'}
    supports_dynamic_schema: bool = True
    supports_catalog: bool = True
    supports_dynamic_catalog: bool = True
    column_type_mappings: Tuple[Tuple[TPattern[str], TypeEngine, GenericDataType], ...] = (
        (re.compile('^tinyint', re.IGNORECASE), TINYINT(), GenericDataType.NUMERIC),
        (re.compile('^largeint', re.IGNORECASE), LARGEINT(), GenericDataType.NUMERIC),
        (re.compile('^decimal.*', re.IGNORECASE), types.DECIMAL(), GenericDataType.NUMERIC),
        (re.compile('^double', re.IGNORECASE), DOUBLE(), GenericDataType.NUMERIC),
        (re.compile('^varchar(\\((\\d+)\\))*$', re.IGNORECASE), types.VARCHAR(), GenericDataType.STRING),
        (re.compile('^char(\\((\\d+)\\))*$', re.IGNORECASE), types.CHAR(), GenericDataType.STRING),
        (re.compile('^json.*', re.IGNORECASE), types.JSON(), GenericDataType.STRING),
        (re.compile('^binary.*', re.IGNORECASE), types.BINARY(), GenericDataType.STRING),
        (re.compile('^quantile_state', re.IGNORECASE), QuantileState(), GenericDataType.STRING),
        (re.compile('^agg_state.*', re.IGNORECASE), AggState(), GenericDataType.STRING),
        (re.compile('^hll', re.IGNORECASE), HLL(), GenericDataType.STRING),
        (re.compile('^bitmap', re.IGNORECASE), BITMAP(), GenericDataType.STRING),
        (re.compile('^array.*', re.IGNORECASE), ARRAY(), GenericDataType.STRING),
        (re.compile('^map.*', re.IGNORECASE), MAP(), GenericDataType.STRING),
        (re.compile('^struct.*', re.IGNORECASE), STRUCT(), GenericDataType.STRING),
        (re.compile('^datetime.*', re.IGNORECASE), types.DATETIME(), GenericDataType.STRING),
        (re.compile('^date.*', re.IGNORECASE), types.DATE(), GenericDataType.STRING),
        (re.compile('^text.*', re.IGNORECASE), TEXT(), GenericDataType.STRING),
        (re.compile('^string.*', re.IGNORECASE), String(), GenericDataType.STRING),
    )
    custom_errors: Dict[
        TPattern[str],
        Tuple[str, SupersetErrorType, Dict[str, List[str]]],
    ] = {
        CONNECTION_ACCESS_DENIED_REGEX: (
            __('Either the username "%(username)s" or the password is incorrect.'),
            SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR,
            {'invalid': ['username', 'password']},
        ),
        CONNECTION_INVALID_HOSTNAME_REGEX: (
            __('Unknown Doris server host "%(hostname)s".'),
            SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR,
            {'invalid': ['host']},
        ),
        CONNECTION_HOST_DOWN_REGEX: (
            __('The host "%(hostname)s" might be down and can\'t be reached.'),
            SupersetErrorType.CONNECTION_HOST_DOWN_ERROR,
            {'invalid': ['host', 'port']},
        ),
        CONNECTION_UNKNOWN_DATABASE_REGEX: (
            __('Unable to connect to database "%(database)s".'),
            SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR,
            {'invalid': ['database']},
        ),
        SYNTAX_ERROR_REGEX: (
            __('Please check your query for syntax errors near "%(server_error)s". Then, try running your query again.'),
            SupersetErrorType.SYNTAX_ERROR,
            {},
        ),
    }

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Tuple[URL, Dict[str, Any]]:
        if catalog:
            pass
        elif uri.database and '.' in uri.database:
            catalog, _ = uri.database.split('.', 1)
        else:
            catalog = 'internal'
        schema = schema or 'information_schema'
        database = '.'.join([catalog or '', schema])
        uri = uri.set(database=database)
        return (uri, connect_args)

    @classmethod
    def get_default_catalog(cls, database: Database) -> Optional[str]:
        """
        Return the default catalog.
        """
        if database.url_object.database is None:
            return None
        return database.url_object.database.split('.')[0]

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> Set[str]:
        """
        Get all catalogs.
        For Doris, the SHOW CATALOGS command returns multiple columns:
        CatalogId, CatalogName, Type, IsCurrent, CreateTime, LastUpdateTime, Comment
        We need to extract just the CatalogName column.
        """
        result = inspector.bind.execute('SHOW CATALOGS')
        return {row.CatalogName for row in result}

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return the configured schema.

        For doris the SQLAlchemy URI looks like this:

            doris://localhost:9030/catalog.database

        """
        database = sqlalchemy_uri.database.strip('/')
        if '.' not in database:
            return None
        return parse.unquote(database.split('.')[1])