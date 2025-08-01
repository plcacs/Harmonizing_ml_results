from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from flask import current_app
from flask_babel import lazy_gettext as _
from marshmallow import EXCLUDE, fields, post_load, pre_load, Schema, validates, validates_schema
from marshmallow.validate import Length, OneOf, Range, ValidationError
from sqlalchemy import MetaData
from werkzeug.datastructures import FileStorage
from superset import db, is_feature_enabled
from superset.commands.database.exceptions import DatabaseInvalidError
from superset.commands.database.ssh_tunnel.exceptions import SSHTunnelingNotEnabledError, SSHTunnelInvalidCredentials, SSHTunnelMissingCredentials
from superset.commands.database.uploaders.base import UploadFileType
from superset.constants import PASSWORD_MASK
from superset.databases.types import EncryptedDict, EncryptedField, EncryptedString
from superset.databases.utils import make_url_safe
from superset.db_engine_specs import get_engine_spec
from superset.exceptions import CertificateException, SupersetSecurityException
from superset.models.core import ConfigurationMethod, Database
from superset.security.analytics_db_safety import check_sqlalchemy_uri
from superset.utils import json
from superset.utils.core import markdown, parse_ssl_cert

database_schemas_query_schema: Dict[str, Any] = {
    'type': 'object',
    'properties': {
        'force': {'type': 'boolean'},
        'catalog': {'type': 'string'}
    }
}
database_catalogs_query_schema: Dict[str, Any] = {
    'type': 'object',
    'properties': {
        'force': {'type': 'boolean'}
    }
}
database_tables_query_schema: Dict[str, Any] = {
    'type': 'object',
    'properties': {
        'force': {'type': 'boolean'},
        'schema_name': {'type': 'string'},
        'catalog_name': {'type': 'string'}
    },
    'required': ['schema_name']
}
database_name_description: str = 'A database name to identify this connection.'
port_description: str = 'Port number for the database connection.'
cache_timeout_description: str = (
    'Duration (in seconds) of the caching timeout for charts of this database. A timeout of 0 indicates that the cache never expires. '
    'Note this defaults to the global timeout if undefined.'
)
expose_in_sqllab_description: str = 'Expose this database to SQLLab'
allow_run_async_description: str = (
    'Operate the database in asynchronous mode, meaning that the queries are executed on remote workers as opposed to on the web server itself. '
    'This assumes that you have a Celery worker setup as well as a results backend. Refer to the installation docs for more information.'
)
allow_file_upload_description: str = (
    'Allow to upload CSV file data into this databaseIf selected, please set the schemas allowed for csv upload in Extra.'
)
allow_ctas_description: str = 'Allow CREATE TABLE AS option in SQL Lab'
allow_cvas_description: str = 'Allow CREATE VIEW AS option in SQL Lab'
allow_dml_description: str = 'Allow users to run non-SELECT statements (UPDATE, DELETE, CREATE, ...) in SQL Lab'
configuration_method_description: str = (
    'Configuration_method is used on the frontend to inform the backend whether to explode parameters or to provide only a sqlalchemy_uri.'
)
impersonate_user_description: str = (
    'If Presto, all the queries in SQL Lab are going to be executed as the currently logged on user who must have permission to run them.'
    '<br/>If Hive and hive.server2.enable.doAs is enabled, will run the queries as service account, but impersonate the currently logged on user via hive.server2.proxy.user property.'
)
force_ctas_schema_description: str = 'When allowing CREATE TABLE AS option in SQL Lab, this option forces the table to be created in this schema'
encrypted_extra_description: str = markdown(
    'JSON string containing additional connection configuration.<br/>This is used to provide connection information for systems like Hive, Presto, and BigQuery, '
    'which do not conform to the username:password syntax normally used by SQLAlchemy.',
    True
)
extra_description: str = markdown(
    'JSON string containing extra configuration elements.<br/>1. The ``engine_params`` object gets unpacked into the '
    '[sqlalchemy.create_engine](https://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine) call, while the '
    '``metadata_params`` gets unpacked into the [sqlalchemy.MetaData](https://docs.sqlalchemy.org/en/rel_1_0/core/metadata.html#sqlalchemy.schema.MetaData) call.'
    '<br/>2. The ``metadata_cache_timeout`` is a cache timeout setting in seconds for metadata fetch of this database. Specify it as **"metadata_cache_timeout": {"schema_cache_timeout": 600, "table_cache_timeout": 600}**. '
    'If unset, cache will not be enabled for the functionality. A timeout of 0 indicates that the cache never expires.'
    '<br/>3. The ``schemas_allowed_for_file_upload`` is a comma separated list of schemas that CSVs are allowed to upload to. Specify it as '
    '**"schemas_allowed_for_file_upload": ["public", "csv_upload"]**. If database flavor does not support schema or any schema is allowed to be accessed, just leave the list empty'
    '<br/>4. The ``version`` field is a string specifying the this db\'s version. This should be used with Presto DBs so that the syntax is correct'
    '<br/>5. The ``allows_virtual_table_explore`` field is a boolean specifying whether or not the Explore button in SQL Lab results is shown.'
    '<br/>6. The ``disable_data_preview`` field is a boolean specifying whether or not data preview queries will be run when fetching table metadata in SQL Lab.'
    '7. The ``disable_drill_to_detail`` field is a boolean specifying whether or notdrill to detail is disabled for the database.'
    '8. The ``allow_multi_catalog`` indicates if the database allows changing the default catalog when running queries and creating datasets.',
    True
)
get_export_ids_schema: Dict[str, Any] = {'type': 'array', 'items': {'type': 'integer'}}
sqlalchemy_uri_description: str = markdown(
    'Refer to the [SqlAlchemy docs](https://docs.sqlalchemy.org/en/rel_1_2/core/engines.html#database-urls) for more information on how to structure your URI.',
    True
)
server_cert_description: str = markdown(
    'Optional CA_BUNDLE contents to validate HTTPS requests. Only available on certain database engines.',
    True
)
openapi_spec_methods_override: Dict[str, Any] = {
    'get_list': {
        'get': {
            'summary': 'Get a list of databases',
            'description': 'Gets a list of databases, use Rison or JSON query parameters for filtering, sorting, pagination and  for selecting specific columns and metadata.'
        }
    },
    'info': {
        'get': {
            'summary': 'Get metadata information about this API resource'
        }
    }
}


def sqlalchemy_uri_validator(value: str) -> str:
    """
    Validate if it's a valid SQLAlchemy URI and refuse SQLLite by default
    """
    try:
        uri = make_url_safe(value.strip())
    except DatabaseInvalidError as ex:
        raise ValidationError([_('Invalid connection string, a valid string usually follows: backend+driver://user:password@database-host/database-name')]) from ex
    if current_app.config.get('PREVENT_UNSAFE_DB_CONNECTIONS', True):
        try:
            check_sqlalchemy_uri(uri)
        except SupersetSecurityException as ex:
            raise ValidationError([str(ex)]) from ex
    return value


def server_cert_validator(value: Optional[str]) -> Optional[str]:
    """
    Validate the server certificate
    """
    if value:
        try:
            parse_ssl_cert(value)
        except CertificateException as ex:
            raise ValidationError([_('Invalid certificate')]) from ex
    return value


def encrypted_extra_validator(value: Optional[str]) -> Optional[str]:
    """
    Validate that encrypted extra is a valid JSON string
    """
    if value:
        try:
            json.loads(value)
        except json.JSONDecodeError as ex:
            raise ValidationError([_('Field cannot be decoded by JSON. %(msg)s', msg=str(ex))]) from ex
    return value


def extra_validator(value: Optional[str]) -> Optional[str]:
    """
    Validate that extra is a valid JSON string, and that metadata_params
    keys are on the call signature for SQLAlchemy Metadata
    """
    if value:
        try:
            extra_ = json.loads(value)
        except json.JSONDecodeError as ex:
            raise ValidationError([_('Field cannot be decoded by JSON. %(msg)s', msg=str(ex))]) from ex
        metadata_signature = inspect.signature(MetaData)
        for key in extra_.get('metadata_params', {}):
            if key not in metadata_signature.parameters:
                raise ValidationError([_('The metadata_params in Extra field is not configured correctly. The key %(key)s is invalid.', key=key)])
    return value


class DatabaseParametersSchemaMixin:
    """
    Allow SQLAlchemy URI to be passed as separate parameters.

    This mixin is a first step in allowing the users to test, create and
    edit databases without having to know how to write a SQLAlchemy URI.
    Instead, each database defines the parameters that it takes (eg,
    username, password, host, etc.) and the SQLAlchemy URI is built from
    these parameters.

    When using this mixin make sure that `sqlalchemy_uri` is not required.
    """
    engine = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy engine to use'})
    driver = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy driver to use'})
    parameters = fields.Dict(keys=fields.String(), values=fields.Raw(), metadata={'description': 'DB-specific parameters for configuration'})
    configuration_method = fields.Enum(ConfigurationMethod, by_value=True, metadata={'description': configuration_method_description}, load_default=ConfigurationMethod.SQLALCHEMY_FORM)

    @pre_load
    def build_sqlalchemy_uri(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Build SQLAlchemy URI from separate parameters.

        This is used for databases that support being configured by individual
        parameters (eg, username, password, host, etc.), instead of requiring
        the constructed SQLAlchemy URI to be passed.
        """
        parameters: Dict[str, Any] = data.pop('parameters', {})
        engine: Optional[str] = data.pop('engine', None) or parameters.pop('engine', None) or data.pop('backend', None)
        driver: Optional[str] = data.pop('driver', None)
        configuration_method = data.get('configuration_method')
        if configuration_method == ConfigurationMethod.DYNAMIC_FORM:
            if not engine:
                raise ValidationError([_('An engine must be specified when passing individual parameters to a database.')])
            engine_spec = get_engine_spec(engine, driver)
            if not hasattr(engine_spec, 'build_sqlalchemy_uri') or not hasattr(engine_spec, 'parameters_schema'):
                raise ValidationError([_('Engine spec "InvalidEngine" does not support being configured via individual parameters.')])
            parameters = engine_spec.parameters_schema.load(parameters)
            serialized_encrypted_extra: str = data.get('masked_encrypted_extra') or '{}'
            try:
                encrypted_extra = json.loads(serialized_encrypted_extra)
            except json.JSONDecodeError:
                encrypted_extra = {}
            data['sqlalchemy_uri'] = engine_spec.build_sqlalchemy_uri(parameters, encrypted_extra)
        return data


def rename_encrypted_extra(self: Any, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """
    Rename ``encrypted_extra`` to ``masked_encrypted_extra``.

    PR #21248 changed the database schema for security reasons. This pre-loader keeps
    Superset backwards compatible with older clients.
    """
    if 'encrypted_extra' in data:
        data['masked_encrypted_extra'] = data.pop('encrypted_extra')
    return data


class DatabaseValidateParametersSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    id = fields.Integer(allow_none=True, metadata={'description': 'Database ID (for updates)'})
    engine = fields.String(required=True, metadata={'description': 'SQLAlchemy engine to use'})
    driver = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy driver to use'})
    parameters = fields.Dict(keys=fields.String(), values=fields.Raw(allow_none=True), metadata={'description': 'DB-specific parameters for configuration'})
    catalog = fields.Dict(keys=fields.String(), values=fields.Raw(allow_none=True), metadata={'description': 'Gsheets specific column for managing label to sheet urls'})
    database_name = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    impersonate_user = fields.Boolean(metadata={'description': impersonate_user_description})
    extra = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    masked_encrypted_extra = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    server_cert = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    configuration_method = fields.Enum(ConfigurationMethod, by_value=True, required=True, metadata={'description': configuration_method_description})


class DatabaseSSHTunnel(Schema):
    id = fields.Integer(allow_none=True, metadata={'description': 'SSH Tunnel ID (for updates)'})
    server_address = fields.String()
    server_port = fields.Integer()
    username = fields.String()
    password = fields.String(required=False)
    private_key = fields.String(required=False)
    private_key_password = fields.String(required=False)


class DatabasePostSchema(DatabaseParametersSchemaMixin, Schema):
    class Meta:
        unknown = EXCLUDE

    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    database_name = fields.String(metadata={'description': database_name_description}, required=True, validate=Length(1, 250))
    cache_timeout = fields.Integer(metadata={'description': cache_timeout_description}, allow_none=True)
    expose_in_sqllab = fields.Boolean(metadata={'description': expose_in_sqllab_description})
    allow_run_async = fields.Boolean(metadata={'description': allow_run_async_description})
    allow_file_upload = fields.Boolean(metadata={'description': allow_file_upload_description})
    allow_ctas = fields.Boolean(metadata={'description': allow_ctas_description})
    allow_cvas = fields.Boolean(metadata={'description': allow_cvas_description})
    allow_dml = fields.Boolean(metadata={'description': allow_dml_description})
    force_ctas_schema = fields.String(metadata={'description': force_ctas_schema_description}, allow_none=True, validate=Length(0, 250))
    impersonate_user = fields.Boolean(metadata={'description': impersonate_user_description})
    masked_encrypted_extra = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    extra = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    server_cert = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    sqlalchemy_uri = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(1, 1024), sqlalchemy_uri_validator])
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    external_url = fields.String(allow_none=True)
    uuid = fields.String(required=False)
    ssh_tunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)


class DatabasePutSchema(DatabaseParametersSchemaMixin, Schema):
    class Meta:
        unknown = EXCLUDE

    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    database_name = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    cache_timeout = fields.Integer(metadata={'description': cache_timeout_description}, allow_none=True)
    expose_in_sqllab = fields.Boolean(metadata={'description': expose_in_sqllab_description})
    allow_run_async = fields.Boolean(metadata={'description': allow_run_async_description})
    allow_file_upload = fields.Boolean(metadata={'description': allow_file_upload_description})
    allow_ctas = fields.Boolean(metadata={'description': allow_ctas_description})
    allow_cvas = fields.Boolean(metadata={'description': allow_cvas_description})
    allow_dml = fields.Boolean(metadata={'description': allow_dml_description})
    force_ctas_schema = fields.String(metadata={'description': force_ctas_schema_description}, allow_none=True, validate=Length(0, 250))
    impersonate_user = fields.Boolean(metadata={'description': impersonate_user_description})
    masked_encrypted_extra = fields.String(metadata={'description': encrypted_extra_description}, allow_none=True, validate=encrypted_extra_validator)
    extra = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    server_cert = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    sqlalchemy_uri = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(0, 1024), sqlalchemy_uri_validator])
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    external_url = fields.String(allow_none=True)
    ssh_tunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)
    uuid = fields.String(required=False)


class DatabaseTestConnectionSchema(DatabaseParametersSchemaMixin, Schema):
    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    database_name = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    impersonate_user = fields.Boolean(metadata={'description': impersonate_user_description})
    extra = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    masked_encrypted_extra = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    server_cert = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    sqlalchemy_uri = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(1, 1024), sqlalchemy_uri_validator])
    ssh_tunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)


class TableMetadataOptionsResponse(TypedDict):
    pass


class TableMetadataColumnsResponse(TypedDict, total=False):
    pass


class TableMetadataForeignKeysIndexesResponse(TypedDict):
    pass


class TableMetadataPrimaryKeyResponse(TypedDict):
    pass


class TableMetadataResponse(TypedDict):
    pass


class TableMetadataOptionsResponseSchema(Schema):
    deferrable = fields.Bool()
    initially = fields.Bool()
    match = fields.Bool()
    ondelete = fields.Bool()
    onupdate = fields.Bool()


class TableMetadataColumnsResponseSchema(Schema):
    keys = fields.List(fields.String(), metadata={'description': ''})
    longType = fields.String(metadata={'description': 'The actual backend long type for the column'})
    name = fields.String(metadata={'description': 'The column name'})
    type = fields.String(metadata={'description': 'The column type'})
    duplicates_constraint = fields.String(required=False)


class TableMetadataForeignKeysIndexesResponseSchema(Schema):
    column_names = fields.List(fields.String(metadata={'description': 'A list of column names that compose the foreign key or  index'}))
    name = fields.String(metadata={'description': 'The name of the foreign key or index'})
    options = fields.Nested(TableMetadataOptionsResponseSchema)
    referred_columns = fields.List(fields.String())
    referred_schema = fields.String()
    referred_table = fields.String()
    type = fields.String()


class TableMetadataPrimaryKeyResponseSchema(Schema):
    column_names = fields.List(fields.String(metadata={'description': 'A list of column names that compose the primary key'}))
    name = fields.String(metadata={'description': 'The primary key index name'})
    type = fields.String()


class TableMetadataResponseSchema(Schema):
    name = fields.String(metadata={'description': 'The name of the table'})
    columns = fields.List(fields.Nested(TableMetadataColumnsResponseSchema), metadata={'description': 'A list of columns and their metadata'})
    foreignKeys = fields.List(fields.Nested(TableMetadataForeignKeysIndexesResponseSchema), metadata={'description': 'A list of foreign keys and their metadata'})
    indexes = fields.List(fields.Nested(TableMetadataForeignKeysIndexesResponseSchema), metadata={'description': 'A list of indexes and their metadata'})
    primaryKey = fields.Nested(TableMetadataPrimaryKeyResponseSchema, metadata={'description': 'Primary keys metadata'})
    selectStar = fields.String(metadata={'description': 'SQL select star'})


class TableExtraMetadataResponseSchema(Schema):
    metadata = fields.Dict()
    partitions = fields.Dict()
    clustering = fields.Dict()


class SelectStarResponseSchema(Schema):
    result = fields.String(metadata={'description': 'SQL select star'})


class SchemasResponseSchema(Schema):
    result = fields.List(fields.String(metadata={'description': 'A database schema name'}))


class CatalogsResponseSchema(Schema):
    result = fields.List(fields.String(metadata={'description': 'A database catalog name'}))


class DatabaseTablesResponse(Schema):
    extra = fields.Dict(metadata={'description': 'Extra data used to specify column metadata'})
    type = fields.String(metadata={'description': 'table or view'})
    value = fields.String(metadata={'description': 'The table or view name'})


class ValidateSQLRequest(Schema):
    sql = fields.String(required=True, metadata={'description': 'SQL statement to validate'})
    catalog = fields.String(required=False, allow_none=True)
    schema = fields.String(required=False, allow_none=True)
    template_params = fields.Dict(required=False, allow_none=True)


class ValidateSQLResponse(Schema):
    line_number = fields.Integer()
    start_column = fields.Integer()
    end_column = fields.Integer()
    message = fields.String()


class DatabaseRelatedChart(Schema):
    id = fields.Integer()
    slice_name = fields.String()
    viz_type = fields.String()


class DatabaseRelatedDashboard(Schema):
    id = fields.Integer()
    json_metadata = fields.Dict()
    slug = fields.String()
    title = fields.String()


class DatabaseRelatedCharts(Schema):
    count = fields.Integer(metadata={'description': 'Chart count'})
    result = fields.List(fields.Nested(DatabaseRelatedChart), metadata={'description': 'A list of dashboards'})


class DatabaseRelatedDashboards(Schema):
    count = fields.Integer(metadata={'description': 'Dashboard count'})
    result = fields.List(fields.Nested(DatabaseRelatedDashboard), metadata={'description': 'A list of dashboards'})


class DatabaseRelatedObjectsResponse(Schema):
    charts = fields.Nested(DatabaseRelatedCharts)
    dashboards = fields.Nested(DatabaseRelatedDashboards)


class DatabaseFunctionNamesResponse(Schema):
    function_names = fields.List(fields.String())


class ImportV1DatabaseExtraSchema(Schema):
    @pre_load
    def fix_schemas_allowed_for_csv_upload(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Fixes for ``schemas_allowed_for_csv_upload``.
        """
        if 'schemas_allowed_for_file_upload' in data:
            data['schemas_allowed_for_csv_upload'] = data.pop('schemas_allowed_for_file_upload')
        schemas_allowed_for_csv_upload = data.get('schemas_allowed_for_csv_upload')
        if isinstance(schemas_allowed_for_csv_upload, str):
            data['schemas_allowed_for_csv_upload'] = json.loads(schemas_allowed_for_csv_upload)
        return data

    metadata_params = fields.Dict(keys=fields.Str(), values=fields.Raw())
    engine_params = fields.Dict(keys=fields.Str(), values=fields.Raw())
    metadata_cache_timeout = fields.Dict(keys=fields.Str(), values=fields.Integer())
    schemas_allowed_for_csv_upload = fields.List(fields.String())
    cost_estimate_enabled = fields.Boolean()
    allows_virtual_table_explore = fields.Boolean(required=False)
    cancel_query_on_windows_unload = fields.Boolean(required=False)
    disable_data_preview = fields.Boolean(required=False)
    disable_drill_to_detail = fields.Boolean(required=False)
    allow_multi_catalog = fields.Boolean(required=False)
    version = fields.String(required=False, allow_none=True)


class ImportV1DatabaseSchema(Schema):
    @pre_load
    def fix_allow_csv_upload(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Fix for ``allow_csv_upload`` .
        """
        if 'allow_file_upload' in data:
            data['allow_csv_upload'] = data.pop('allow_file_upload')
        return data

    database_name = fields.String(required=True)
    sqlalchemy_uri = fields.String(required=True)
    password = fields.String(allow_none=True)
    cache_timeout = fields.Integer(allow_none=True)
    expose_in_sqllab = fields.Boolean()
    allow_run_async = fields.Boolean()
    allow_ctas = fields.Boolean()
    allow_cvas = fields.Boolean()
    allow_dml = fields.Boolean(required=False)
    allow_csv_upload = fields.Boolean()
    impersonate_user = fields.Boolean()
    extra = fields.Nested(ImportV1DatabaseExtraSchema)
    uuid = fields.UUID(required=True)
    version = fields.String(required=True)
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    external_url = fields.String(allow_none=True)
    ssh_tunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)

    @validates_schema
    def validate_password(self, data: Dict[str, Any], **kwargs: Any) -> None:
        """If sqlalchemy_uri has a masked password, password is required"""
        uuid_value = data['uuid']
        existing = db.session.query(Database).filter_by(uuid=uuid_value).first()
        if existing:
            return
        uri = data['sqlalchemy_uri']
        password_in_uri = make_url_safe(uri).password
        if password_in_uri == PASSWORD_MASK and data.get('password') is None:
            raise ValidationError('Must provide a password for the database')

    @validates_schema
    def validate_ssh_tunnel_credentials(self, data: Dict[str, Any], **kwargs: Any) -> None:
        """If ssh_tunnel has a masked credentials, credentials are required"""
        uuid_value = data['uuid']
        existing = db.session.query(Database).filter_by(uuid=uuid_value).first()
        if existing:
            return
        ssh_tunnel = data.get('ssh_tunnel')
        if ssh_tunnel:
            if not is_feature_enabled('SSH_TUNNELING'):
                raise SSHTunnelingNotEnabledError()
            password = ssh_tunnel.get('password')
            private_key = ssh_tunnel.get('private_key')
            private_key_password = ssh_tunnel.get('private_key_password')
            if password is not None:
                if private_key is not None or private_key_password is not None:
                    raise SSHTunnelInvalidCredentials()
                if password == PASSWORD_MASK:
                    raise ValidationError('Must provide a password for the ssh tunnel')
            if password is None:
                if private_key is None and private_key_password is None:
                    raise SSHTunnelMissingCredentials()
                exception_messages: List[str] = []
                if private_key is None or private_key == PASSWORD_MASK:
                    exception_messages.append('Must provide a private key for the ssh tunnel')
                if private_key_password is None or private_key_password == PASSWORD_MASK:
                    exception_messages.append('Must provide a private key password for the ssh tunnel')
                if exception_messages:
                    raise ValidationError(exception_messages)
        return


def encrypted_field_properties(self: Any, field: Any, **_) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    if isinstance(field, EncryptedField):
        if self.openapi_version.major > 2:
            ret['x-encrypted-extra'] = True
    return ret


class DatabaseSchemaAccessForFileUploadResponse(Schema):
    schemas = fields.List(fields.String(), metadata={'description': 'The list of schemas allowed for the database to upload information'})


class EngineInformationSchema(Schema):
    supports_file_upload = fields.Boolean(metadata={'description': 'Users can upload files to the database'})
    disable_ssh_tunneling = fields.Boolean(metadata={'description': 'SSH tunnel is not available to the database'})
    supports_dynamic_catalog = fields.Boolean(metadata={'description': 'The database supports multiple catalogs in a single connection'})
    supports_oauth2 = fields.Boolean(metadata={'description': 'The database supports OAuth2'})


class DatabaseConnectionSchema(Schema):
    """
    Schema with database connection information.

    This is only for admins (who have ``can_create`` on ``Database``).
    """
    allow_ctas = fields.Boolean(metadata={'description': allow_ctas_description})
    allow_cvas = fields.Boolean(metadata={'description': allow_cvas_description})
    allow_dml = fields.Boolean(metadata={'description': allow_dml_description})
    allow_file_upload = fields.Boolean(metadata={'description': allow_file_upload_description})
    allow_run_async = fields.Boolean(metadata={'description': allow_run_async_description})
    backend = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy engine to use'})
    cache_timeout = fields.Integer(metadata={'description': cache_timeout_description}, allow_none=True)
    configuration_method = fields.String(metadata={'description': configuration_method_description})
    database_name = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    driver = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy driver to use'})
    engine_information = fields.Nested(EngineInformationSchema)
    expose_in_sqllab = fields.Boolean(metadata={'description': expose_in_sqllab_description})
    extra = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    force_ctas_schema = fields.String(metadata={'description': force_ctas_schema_description}, allow_none=True, validate=Length(0, 250))
    id = fields.Integer(metadata={'description': 'Database ID (for updates)'})
    impersonate_user = fields.Boolean(metadata={'description': impersonate_user_description})
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    server_cert = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    uuid = fields.String(required=False)
    ssh_tunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)
    masked_encrypted_extra = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    parameters = fields.Dict(keys=fields.String(), values=fields.Raw(), metadata={'description': 'DB-specific parameters for configuration'})
    parameters_schema = fields.Dict(keys=fields.String(), values=fields.Raw(), metadata={'description': 'JSONSchema for configuring the database by parameters instead of SQLAlchemy URI'})
    sqlalchemy_uri = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(1, 1024), sqlalchemy_uri_validator])


class DelimitedListField(fields.List):
    """
    Special marshmallow field for handling delimited lists.
    formData expects a string, so we need to deserialize it into a list.
    """

    def _deserialize(self, value: Any, attr: str, data: Any, **kwargs: Any) -> List[Any]:
        try:
            values = value.split(',') if value else []
            return super()._deserialize(values, attr, data, **kwargs)
        except AttributeError as exc:
            raise ValidationError(f'{attr} is not a delimited list it has a non string value {value}.') from exc


class BaseUploadFilePostSchemaMixin(Schema):
    @validates('file')
    def validate_file_extension(self, file: FileStorage) -> None:
        allowed_extensions: List[str] = current_app.config['ALLOWED_EXTENSIONS']
        file_suffix: str = Path(file.filename).suffix
        if not file_suffix or file_suffix[1:] not in allowed_extensions:
            raise ValidationError([_('File extension is not allowed.')])


class UploadPostSchema(BaseUploadFilePostSchemaMixin):
    type = fields.Enum(UploadFileType, required=True, by_value=True, metadata={'description': 'File type to upload'})
    already_exists = fields.String(load_default='fail', validate=OneOf(choices=('fail', 'replace', 'append')), metadata={'description': 'What to do if the table already exists accepts: fail, replace, append'})
    index_label = fields.String(metadata={'description': 'Index label for index column.'})
    columns_read = DelimitedListField(fields.String(), metadata={'description': 'A List of the column names that should be read'})
    dataframe_index = fields.Boolean(metadata={'description': 'Write dataframe index as a column.'})
    schema = fields.String(metadata={'description': 'The schema to upload the data file to.'})
    table_name = fields.String(required=True, validate=[Length(min=1, max=10000)], allow_none=False, metadata={'description': 'The name of the table to be created/appended'})
    file = fields.Raw(required=True, metadata={'description': 'The file to upload', 'type': 'string', 'format': 'text/csv'})
    delimiter = fields.String(metadata={'description': '[CSV only] The character used to separate values in the CSV file (e.g., a comma, semicolon, or tab).'})
    column_data_types = fields.String(metadata={'description': "[CSV only] A dictionary with column names and their data types if you need to change the defaults. Example: {'user_id':'int'}. Check Python Pandas library for supported data types"})
    day_first = fields.Boolean(metadata={'description': '[CSV only] DD/MM format dates, international and European format'})
    skip_blank_lines = fields.Boolean(metadata={'description': '[CSV only] Skip blank lines in the CSV file.'})
    skip_initial_space = fields.Boolean(metadata={'description': '[CSV only] Skip spaces after delimiter.'})
    column_dates = DelimitedListField(fields.String(), metadata={'description': '[CSV and Excel only] A list of column names that should be parsed as dates. Example: date,timestamp'})
    decimal_character = fields.String(metadata={'description': "[CSV and Excel only] Character to recognize as decimal point. Default is '.'"})
    header_row = fields.Integer(metadata={'description': '[CSV and Excel only] Row containing the headers to use as column names (0 is first line of data). Leave empty if there is no header row.'})
    index_column = fields.String(metadata={'description': '[CSV and Excel only] Column to use as the row labels of the dataframe. Leave empty if no index column'})
    null_values = DelimitedListField(fields.String(), metadata={'description': "[CSV and Excel only] A list of strings that should be treated as null. Examples: '' for empty strings, 'None', 'N/A', Warning: Hive database supports only a single value"})
    rows_to_read = fields.Integer(metadata={'description': '[CSV and Excel only] Number of rows to read from the file. If None, reads all rows.'}, allow_none=True, validate=Range(min=1))
    skip_rows = fields.Integer(metadata={'description': '[CSV and Excel only] Number of rows to skip at start of file.'})
    sheet_name = fields.String(metadata={'description': '[Excel only]] Strings used for sheet names (default is the first sheet).'})

    @post_load
    def convert_column_data_types(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if 'column_data_types' in data and data['column_data_types']:
            try:
                data['column_data_types'] = json.loads(data['column_data_types'])
            except json.JSONDecodeError as ex:
                raise ValidationError('Invalid JSON format for column_data_types') from ex
        return data


class UploadFileMetadataPostSchema(BaseUploadFilePostSchemaMixin):
    """
    Schema for Upload file metadata.
    """
    type = fields.Enum(UploadFileType, required=True, by_value=True, metadata={'description': 'File type to upload'})
    file = fields.Raw(required=True, metadata={'description': 'The file to upload', 'type': 'string', 'format': 'binary'})
    delimiter = fields.String(metadata={'description': 'The character used to separate values in the CSV file (e.g., a comma, semicolon, or tab).'})
    header_row = fields.Integer(metadata={'description': 'Row containing the headers to use as column names(0 is first line of data). Leave empty if there is no header row.'})


class UploadFileMetadataItemSchema(Schema):
    sheet_name = fields.String(metadata={'description': 'The name of the sheet'})
    column_names = fields.List(fields.String(), metadata={'description': 'A list of columns names in the sheet'})


class UploadFileMetadata(Schema):
    """
    Schema for upload file metadata response.
    """
    items = fields.List(fields.Nested(UploadFileMetadataItemSchema))


class OAuth2ProviderResponseSchema(Schema):
    """
    Schema for the payload sent on OAuth2 redirect.
    """
    code = fields.String(required=False, metadata={'description': 'The authorization code returned by the provider'})
    state = fields.String(required=False, metadata={'description': 'The state parameter originally passed by the client'})
    scope = fields.String(required=False, metadata={'description': 'A space-separated list of scopes granted by the user'})
    error = fields.String(required=False, metadata={'description': 'In case of an error, this field contains the error code'})
    error_description = fields.String(required=False, metadata={'description': 'Additional description of the error'})

    class Meta:
        unknown = EXCLUDE


class QualifiedTableSchema(Schema):
    """
    Schema for a qualified table reference.

    Catalog and schema can be ommited, to fallback to default values. Table name must be
    present.
    """
    name = fields.String(required=True, metadata={'description': 'The table name'})
    schema = fields.String(required=False, load_default=None, metadata={'description': 'The table schema'})
    catalog = fields.String(required=False, load_default=None, metadata={'description': 'The table catalog'})
