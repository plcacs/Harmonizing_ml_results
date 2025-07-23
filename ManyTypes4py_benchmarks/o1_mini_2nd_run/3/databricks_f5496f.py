from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING
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
    access_token: fields.Str = fields.Str(required=True)
    host: fields.Str = fields.Str(required=True)
    port: fields.Integer = fields.Integer(
        required=True,
        metadata={'description': __('Database port')},
        validate=Range(min=0, max=2 ** 16, max_inclusive=False)
    )
    encryption: fields.Boolean = fields.Boolean(
        required=False,
        metadata={'description': __('Use an encrypted connection to the database')}
    )


class DatabricksBaseParametersType(TypedDict):
    """
    The parameters are all the keys that do not exist on the Database model.
    These are used to build the sqlalchemy uri.
    """


class DatabricksNativeSchema(DatabricksBaseSchema):
    """
    Additional fields required only for the DatabricksNativeEngineSpec.
    """
    database: fields.Str = fields.Str(required=True)


class DatabricksNativePropertiesSchema(DatabricksNativeSchema):
    """
    Properties required only for the DatabricksNativeEngineSpec.
    """
    http_path: fields.Str = fields.Str(required=True)


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
    http_path_field: fields.Str = fields.Str(required=True)
    default_catalog: fields.Str = fields.Str(required=True)
    default_schema: fields.Str = fields.Str(required=True)


class DatabricksPythonConnectorParametersType(DatabricksBaseParametersType):
    """
    Additional parameters required only for the DatabricksPythonConnectorEngineSpec.
    """


class DatabricksPythonConnectorPropertiesType(TypedDict):
    """
    All properties that need to be available to the DatabricksPythonConnectorEngineSpec
    in order to create a connection if the dynamic form is used.
    """


time_grain_expressions: Dict[Optional[TimeGrain], str] = {
    None: '{col}',
    TimeGrain.SECOND: "date_trunc('second', {col})",
    TimeGrain.MINUTE: "date_trunc('minute', {col})",
    TimeGrain.HOUR: "date_trunc('hour', {col})",
    TimeGrain.DAY: "date_trunc('day', {col})",
    TimeGrain.WEEK: "date_trunc('week', {col})",
    TimeGrain.MONTH: "date_trunc('month', {col})",
    TimeGrain.QUARTER: "date_trunc('quarter', {col})",
    TimeGrain.YEAR: "date_trunc('year', {col})",
    TimeGrain.WEEK_ENDING_SATURDAY: "date_trunc('week', {col} + interval '1 day') + interval '5 days'",
    TimeGrain.WEEK_STARTING_SUNDAY: "date_trunc('week', {col} + interval '1 day') - interval '1 day'"
}


class DatabricksHiveEngineSpec(HiveEngineSpec):
    engine_name: str = 'Databricks Interactive Cluster'
    engine: str = 'databricks'
    drivers: Dict[str, str] = {'pyhive': 'Hive driver for Interactive Cluster'}
    default_driver: str = 'pyhive'
    _show_functions_column: str = 'function'
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = time_grain_expressions


class DatabricksBaseEngineSpec(BaseEngineSpec):
    _time_grain_expressions: Dict[Optional[TimeGrain], str] = time_grain_expressions

    @classmethod
    def convert_dttm(
        cls,
        target_type: str,
        dttm: datetime,
        db_extra: Optional[Dict[str, Any]] = None
    ) -> str:
        return HiveEngineSpec.convert_dttm(target_type, dttm, db_extra=db_extra)

    @classmethod
    def epoch_to_dttm(cls) -> str:
        return HiveEngineSpec.epoch_to_dttm()


class DatabricksODBCEngineSpec(DatabricksBaseEngineSpec):
    engine_name: str = 'Databricks SQL Endpoint'
    engine: str = 'databricks'
    drivers: Dict[str, str] = {'pyodbc': 'ODBC driver for SQL endpoint'}
    default_driver: str = 'pyodbc'


class DatabricksDynamicBaseEngineSpec(BasicParametersMixin, DatabricksBaseEngineSpec):
    default_driver: str = ''
    encryption_parameters: Dict[str, str] = {'ssl': '1'}
    required_parameters: Set[str] = {'access_token', 'host', 'port'}
    context_key_mapping: Dict[str, str] = {'access_token': 'password', 'host': 'hostname', 'port': 'port'}

    @staticmethod
    def get_extra_params(database: Database) -> Dict[str, Any]:
        """
        Add a user agent to be used in the requests.
        Trim whitespace from connect_args to avoid databricks driver errors
        """
        extra: Dict[str, Any] = BaseEngineSpec.get_extra_params(database)
        engine_params: Dict[str, Any] = extra.setdefault('engine_params', {})
        connect_args: Dict[str, Any] = engine_params.setdefault('connect_args', {})
        connect_args.setdefault('http_headers', [('User-Agent', USER_AGENT)])
        connect_args.setdefault('_user_agent_entry', USER_AGENT)
        if (http_path := connect_args.get('http_path')) is not None:
            connect_args['http_path'] = http_path.strip()
        return extra

    @classmethod
    def get_table_names(
        cls,
        database: Database,
        inspector: Inspector,
        schema: str
    ) -> Set[str]:
        return super().get_table_names(database, inspector, schema) - cls.get_view_names(database, inspector, schema)

    @classmethod
    def extract_errors(
        cls,
        ex: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SupersetError]:
        raw_message: Optional[str] = cls._extract_error_message(ex)
        context = context or {}
        for key, value in cls.context_key_mapping.items():
            context[key] = context.get(value)
        for regex, (message, error_type, extra) in cls.custom_errors.items():
            match = regex.search(raw_message) if raw_message else None
            if match:
                params: Dict[str, Any] = {**context, **match.groupdict()}
                extra['engine_name'] = cls.engine_name
                return [
                    SupersetError(
                        error_type=error_type,
                        message=message % params,
                        level=ErrorLevel.ERROR,
                        extra=extra
                    )
                ]
        return [
            SupersetError(
                error_type=SupersetErrorType.GENERIC_DB_ENGINE_ERROR,
                message=cls._extract_error_message(ex) or '',
                level=ErrorLevel.ERROR,
                extra={'engine_name': cls.engine_name}
            )
        ]

    @classmethod
    def validate_parameters(cls, properties: Dict[str, Any]) -> List[SupersetError]:
        errors: List[SupersetError] = []
        extra: Optional[Dict[str, Any]] = json.loads(properties.get('extra') or '{}')
        if extra:
            engine_params: Dict[str, Any] = extra.get('engine_params', {})
            connect_args: Dict[str, Any] = engine_params.get('connect_args', {})
        else:
            engine_params = {}
            connect_args = {}
        parameters: Dict[str, Any] = {**properties, **properties.get('parameters', {})}
        if connect_args.get('http_path'):
            parameters['http_path'] = connect_args.get('http_path')
        present: Set[str] = {key for key in parameters if parameters.get(key, ())}
        missing: List[str] = sorted(cls.required_parameters - present)
        if missing:
            errors.append(
                SupersetError(
                    message=f'One or more parameters are missing: {", ".join(missing)}',
                    error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR,
                    level=ErrorLevel.WARNING,
                    extra={'missing': missing}
                )
            )
        host: Optional[str] = parameters.get('host')
        if not host:
            return errors
        if not is_hostname_valid(host):
            errors.append(
                SupersetError(
                    message="The hostname provided can't be resolved.",
                    error_type=SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR,
                    level=ErrorLevel.ERROR,
                    extra={'invalid': ['host']}
                )
            )
            return errors
        port: Optional[Union[int, str]] = parameters.get('port')
        if not port:
            return errors
        try:
            port = int(port)
        except (ValueError, TypeError):
            errors.append(
                SupersetError(
                    message='Port must be a valid integer.',
                    error_type=SupersetErrorType.CONNECTION_INVALID_PORT_ERROR,
                    level=ErrorLevel.ERROR,
                    extra={'invalid': ['port']}
                )
            )
            return errors
        if not (0 <= port < 2 ** 16):
            errors.append(
                SupersetError(
                    message='The port must be an integer between 0 and 65535 (inclusive).',
                    error_type=SupersetErrorType.CONNECTION_INVALID_PORT_ERROR,
                    level=ErrorLevel.ERROR,
                    extra={'invalid': ['port']}
                )
            )
        elif not is_port_open(host, port):
            errors.append(
                SupersetError(
                    message='The port is closed.',
                    error_type=SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR,
                    level=ErrorLevel.ERROR,
                    extra={'invalid': ['port']}
                )
            )
        return errors


class DatabricksNativeEngineSpec(DatabricksDynamicBaseEngineSpec):
    engine: str = 'databricks'
    engine_name: str = 'Databricks'
    drivers: Dict[str, str] = {'connector': 'Native all-purpose driver'}
    default_driver: str = 'connector'
    parameters_schema: DatabricksNativeSchema = DatabricksNativeSchema()
    properties_schema: DatabricksNativePropertiesSchema = DatabricksNativePropertiesSchema()
    sqlalchemy_uri_placeholder: str = 'databricks+connector://token:{access_token}@{host}:{port}/{database_name}'
    context_key_mapping: Dict[str, str] = {
        **DatabricksDynamicBaseEngineSpec.context_key_mapping,
        'database': 'database',
        'username': 'username'
    }
    required_parameters: Set[str] = DatabricksDynamicBaseEngineSpec.required_parameters | {'database', 'extra'}
    supports_dynamic_schema: bool = True
    supports_catalog: bool = True
    supports_dynamic_catalog: bool = True

    @classmethod
    def build_sqlalchemy_uri(
        cls,
        parameters: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ) -> str:
        query: Dict[str, str] = {}
        if parameters.get('encryption'):
            if not cls.encryption_parameters:
                raise Exception('Unable to build a URL with encryption enabled')
            query.update(cls.encryption_parameters)
        return str(
            URL.create(
                f'{cls.engine}+{cls.default_driver}'.rstrip('+'),
                username='token',
                password=parameters.get('access_token'),
                host=parameters['host'],
                port=parameters['port'],
                database=parameters['database'],
                query=query
            )
        )

    @classmethod
    def get_parameters_from_uri(
        cls,
        uri: str,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        url: URL = make_url_safe(uri)
        encryption: bool = all(
            (item in url.query.items() for item in cls.encryption_parameters.items())
        )
        return {
            'access_token': url.password,
            'host': url.host,
            'port': url.port,
            'database': url.database,
            'encryption': encryption
        }

    @classmethod
    def parameters_json_schema(cls) -> Optional[Dict[str, Any]]:
        """
        Return configuration parameters as OpenAPI.
        """
        if not cls.properties_schema:
            return None
        spec: APISpec = APISpec(
            title='Database Parameters',
            version='1.0.0',
            openapi_version='3.0.2',
            plugins=[MarshmallowPlugin()]
        )
        spec.components.schema(cls.__name__, schema=cls.properties_schema)
        return spec.to_dict()['components']['schemas'].get(cls.__name__)

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        """
        Return the default catalog.

        The default behavior for Databricks is confusing. When Unity Catalog is not
        enabled we have (the DB engine spec hasn't been tested with it enabled):

            > SHOW CATALOGS;
            spark_catalog
            > SELECT current_catalog();
            hive_metastore

        To handle permissions correctly we use the result of `SHOW CATALOGS` when a
        single catalog is returned.
        """
        with database.get_sqla_engine() as engine:
            catalogs: Set[str] = {catalog for (catalog,) in engine.execute('SHOW CATALOGS')}
            if len(catalogs) == 1:
                return catalogs.pop()
            result: Any = engine.execute('SELECT current_catalog()').scalar()
            return result if result is not None else ''

    @classmethod
    def get_prequeries(
        cls,
        database: Database,
        catalog: Optional[str] = None,
        schema: Optional[str] = None
    ) -> List[str]:
        prequeries: List[str] = []
        if catalog:
            catalog_formatted: str = f'`{catalog}`' if not catalog.startswith('`') else catalog
            prequeries.append(f'USE CATALOG {catalog_formatted}')
        if schema:
            schema_formatted: str = f'`{schema}`' if not schema.startswith('`') else schema
            prequeries.append(f'USE SCHEMA {schema_formatted}')
        return prequeries

    @classmethod
    def get_catalog_names(
        cls,
        database: Database,
        inspector: Inspector
    ) -> Set[str]:
        return {catalog for (catalog,) in inspector.bind.execute('SHOW CATALOGS')}


class DatabricksPythonConnectorEngineSpec(DatabricksDynamicBaseEngineSpec):
    engine: str = 'databricks'
    engine_name: str = 'Databricks Python Connector'
    default_driver: str = 'databricks-sql-python'
    drivers: Dict[str, str] = {'databricks-sql-python': 'Databricks SQL Python'}
    parameters_schema: DatabricksPythonConnectorSchema = DatabricksPythonConnectorSchema()
    sqlalchemy_uri_placeholder: str = 'databricks://token:{access_token}@{host}:{port}?http_path={http_path}&catalog={default_catalog}&schema={default_schema}'
    context_key_mapping: Dict[str, str] = {
        **DatabricksDynamicBaseEngineSpec.context_key_mapping,
        'default_catalog': 'catalog',
        'default_schema': 'schema',
        'http_path_field': 'http_path'
    }
    required_parameters: Set[str] = DatabricksDynamicBaseEngineSpec.required_parameters | {'default_catalog', 'default_schema', 'http_path_field'}
    supports_dynamic_schema: bool = True
    supports_catalog: bool = True
    supports_dynamic_catalog: bool = True

    @classmethod
    def build_sqlalchemy_uri(
        cls,
        parameters: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ) -> str:
        query: Dict[str, str] = {}
        if (http_path := parameters.get('http_path_field')) is not None:
            query['http_path'] = http_path
        if (catalog := parameters.get('default_catalog')) is not None:
            query['catalog'] = catalog
        if (schema := parameters.get('default_schema')) is not None:
            query['schema'] = schema
        if parameters.get('encryption'):
            query.update(cls.encryption_parameters)
        return str(
            URL.create(
                cls.engine,
                username='token',
                password=parameters.get('access_token'),
                host=parameters['host'],
                port=parameters['port'],
                query=query
            )
        )

    @classmethod
    def get_parameters_from_uri(
        cls,
        uri: str,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        url: URL = make_url_safe(uri)
        query: Dict[str, str] = {
            key: value for key, value in url.query.items()
            if (key, value) not in cls.encryption_parameters.items()
        }
        encryption: bool = all(
            (item in url.query.items() for item in cls.encryption_parameters.items())
        )
        return {
            'access_token': url.password,
            'host': url.host,
            'port': url.port,
            'http_path_field': query.get('http_path', ''),
            'default_catalog': query.get('catalog', ''),
            'default_schema': query.get('schema', ''),
            'encryption': encryption
        }

    @classmethod
    def get_default_catalog(cls, database: Database) -> Optional[str]:
        return database.url_object.query.get('catalog')

    @classmethod
    def get_catalog_names(
        cls,
        database: Database,
        inspector: Inspector
    ) -> Set[str]:
        return {catalog for (catalog,) in inspector.bind.execute('SHOW CATALOGS')}

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None
    ) -> Tuple[URL, Dict[str, Any]]:
        if catalog:
            uri = uri.update_query_dict({'catalog': catalog})
        if schema:
            uri = uri.update_query_dict({'schema': schema})
        return uri, connect_args
