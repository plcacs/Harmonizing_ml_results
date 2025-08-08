from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, TypedDict
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

database_schemas_query_schema: dict = {'type': 'object', 'properties': {'force': {'type': 'boolean'}, 'catalog': {'type': 'string'}}}
database_catalogs_query_schema: dict = {'type': 'object', 'properties': {'force': {'type': 'boolean'}}}
database_tables_query_schema: dict = {'type': 'object', 'properties': {'force': {'type': 'boolean'}, 'schema_name': {'type': 'string'}, 'catalog_name': {'type': 'string'}}, 'required': ['schema_name']}
database_name_description: str = 'A database name to identify this connection.'
port_description: str = 'Port number for the database connection.'
cache_timeout_description: str = 'Duration (in seconds) of the caching timeout for charts of this database. A timeout of 0 indicates that the cache never expires. Note this defaults to the global timeout if undefined.'
expose_in_sqllab_description: str = 'Expose this database to SQLLab'
allow_run_async_description: str = 'Operate the database in asynchronous mode, meaning that the queries are executed on remote workers as opposed to on the web server itself. This assumes that you have a Celery worker setup as well as a results backend. Refer to the installation docs for more information.'
allow_file_upload_description: str = 'Allow to upload CSV file data into this databaseIf selected, please set the schemas allowed for csv upload in Extra.'
allow_ctas_description: str = 'Allow CREATE TABLE AS option in SQL Lab'
allow_cvas_description: str = 'Allow CREATE VIEW AS option in SQL Lab'
allow_dml_description: str = 'Allow users to run non-SELECT statements (UPDATE, DELETE, CREATE, ...) in SQL Lab'
configuration_method_description: str = 'Configuration_method is used on the frontend to inform the backend whether to explode parameters or to provide only a sqlalchemy_uri.'
impersonate_user_description: str = 'If Presto, all the queries in SQL Lab are going to be executed as the currently logged on user who must have permission to run them.<br/>If Hive and hive.server2.enable.doAs is enabled, will run the queries as service account, but impersonate the currently logged on user via hive.server2.proxy.user property.'
force_ctas_schema_description: str = 'When allowing CREATE TABLE AS option in SQL Lab, this option forces the table to be created in this schema'
encrypted_extra_description: str = markdown('JSON string containing additional connection configuration.<br/>This is used to provide connection information for systems like Hive, Presto, and BigQuery, which do not conform to the username:password syntax normally used by SQLAlchemy.', True)
extra_description: str = markdown('JSON string containing extra configuration elements.<br/>1. The ``engine_params`` object gets unpacked into the [sqlalchemy.create_engine](https://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine) call, while the ``metadata_params`` gets unpacked into the [sqlalchemy.MetaData](https://docs.sqlalchemy.org/rel_1_0/core/metadata.html#sqlalchemy.schema.MetaData) call.<br/>2. The ``metadata_cache_timeout`` is a cache timeout setting in seconds for metadata fetch of this database. Specify it as **"metadata_cache_timeout": {"schema_cache_timeout": 600, "table_cache_timeout": 600}**. If unset, cache will not be enabled for the functionality. A timeout of 0 indicates that the cache never expires.<br/>3. The ``schemas_allowed_for_file_upload`` is a comma separated list of schemas that CSVs are allowed to upload to. Specify it as **"schemas_allowed_for_file_upload": ["public", "csv_upload"]**. If database flavor does not support schema or any schema is allowed to be accessed, just leave the list empty<br/>4. The ``version`` field is a string specifying the this db\'s version. This should be used with Presto DBs so that the syntax is correct<br/>5. The ``allows_virtual_table_explore`` field is a boolean specifying whether or not the Explore button in SQL Lab results is shown.<br/>6. The ``disable_data_preview`` field is a boolean specifying whether or not data preview queries will be run when fetching table metadata in SQL Lab.7. The ``disable_drill_to_detail`` field is a boolean specifying whether or notdrill to detail is disabled for the database.8. The ``allow_multi_catalog`` indicates if the database allows changing the default catalog when running queries and creating datasets.', True)
get_export_ids_schema: dict = {'type': 'array', 'items': {'type': 'integer'}}
sqlalchemy_uri_description: str = markdown('Refer to the [SqlAlchemy docs](https://docs.sqlalchemy.org/en/rel_1_2/core/engines.html#database-urls) for more information on how to structure your URI.', True)
server_cert_description: str = markdown('Optional CA_BUNDLE contents to validate HTTPS requests. Only available on certain database engines.', True)
openapi_spec_methods_override: dict = {'get_list': {'get': {'summary': 'Get a list of databases', 'description': 'Gets a list of databases, use Rison or JSON query parameters for filtering, sorting, pagination and  for selecting specific columns and metadata.'}}, 'info': {'get': {'summary': 'Get metadata information about this API resource'}}}

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

def server_cert_validator(value: str) -> str:
    """
    Validate the server certificate
    """
    if value:
        try:
            parse_ssl_cert(value)
        except CertificateException as ex:
            raise ValidationError([_('Invalid certificate')]) from ex
    return value

def encrypted_extra_validator(value: str) -> str:
    """
    Validate that encrypted extra is a valid JSON string
    """
    if value:
        try:
            json.loads(value)
        except json.JSONDecodeError as ex:
            raise ValidationError([_('Field cannot be decoded by JSON. %(msg)s', msg=str(ex))]) from ex
    return value

def extra_validator(value: str) -> str:
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
    engine: str = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy engine to use'})
    driver: str = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy driver to use'})
    parameters: dict = fields.Dict(keys=fields.String(), values=fields.Raw(), metadata={'description': 'DB-specific parameters for configuration'})
    configuration_method: ConfigurationMethod = fields.Enum(ConfigurationMethod, by_value=True, metadata={'description': configuration_method_description}, load_default=ConfigurationMethod.SQLALCHEMY_FORM)

    @pre_load
    def build_sqlalchemy_uri(self, data: dict, **kwargs: Any) -> dict:
        """
        Build SQLAlchemy URI from separate parameters.

        This is used for databases that support being configured by individual
        parameters (eg, username, password, host, etc.), instead of requiring
        the constructed SQLAlchemy URI to be passed.
        """
        parameters = data.pop('parameters', {})
        engine = data.pop('engine', None) or parameters.pop('engine', None) or data.pop('backend', None)
        driver = data.pop('driver', None)
        configuration_method = data.get('configuration_method')
        if configuration_method == ConfigurationMethod.DYNAMIC_FORM:
            if not engine:
                raise ValidationError([_('An engine must be specified when passing individual parameters to a database.')])
            engine_spec = get_engine_spec(engine, driver)
            if not hasattr(engine_spec, 'build_sqlalchemy_uri') or not hasattr(engine_spec, 'parameters_schema'):
                raise ValidationError([_('Engine spec "InvalidEngine" does not support being configured via individual parameters.')])
            parameters = engine_spec.parameters_schema.load(parameters)
            serialized_encrypted_extra = data.get('masked_encrypted_extra') or '{}'
            try:
                encrypted_extra = json.loads(serialized_encrypted_extra)
            except json.JSONDecodeError:
                encrypted_extra = {}
            data['sqlalchemy_uri'] = engine_spec.build_sqlalchemy_uri(parameters, encrypted_extra)
        return data

def rename_encrypted_extra(self: Any, data: dict, **kwargs: Any) -> dict:
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
    id: int = fields.Integer(allow_none=True, metadata={'description': 'Database ID (for updates)'})
    engine: str = fields.String(required=True, metadata={'description': 'SQLAlchemy engine to use'})
    driver: str = fields.String(allow_none=True, metadata={'description': 'SQLAlchemy driver to use'})
    parameters: dict = fields.Dict(keys=fields.String(), values=fields.Raw(allow_none=True), metadata={'description': 'DB-specific parameters for configuration'})
    catalog: dict = fields.Dict(keys=fields.String(), values=fields.Raw(allow_none=True), metadata={'description': 'Gsheets specific column for managing label to sheet urls'})
    database_name: str = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    impersonate_user: bool = fields.Boolean(metadata={'description': impersonate_user_description})
    extra: str = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    masked_encrypted_extra: str = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    server_cert: str = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    configuration_method: ConfigurationMethod = fields.Enum(ConfigurationMethod, by_value=True, required=True, metadata={'description': configuration_method_description})

class DatabaseSSHTunnel(Schema):

    id: int = fields.Integer(allow_none=True, metadata={'description': 'SSH Tunnel ID (for updates)'})
    server_address: str = fields.String()
    server_port: int = fields.Integer()
    username: str = fields.String()
    password: str = fields.String(required=False)
    private_key: str = fields.String(required=False)
    private_key_password: str = fields.String(required=False)

class DatabasePostSchema(DatabaseParametersSchemaMixin, Schema):

    class Meta:
        unknown = EXCLUDE
    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    database_name: str = fields.String(metadata={'description': database_name_description}, required=True, validate=Length(1, 250))
    cache_timeout: int = fields.Integer(metadata={'description': cache_timeout_description}, allow_none=True)
    expose_in_sqllab: bool = fields.Boolean(metadata={'description': expose_in_sqllab_description})
    allow_run_async: bool = fields.Boolean(metadata={'description': allow_run_async_description})
    allow_file_upload: bool = fields.Boolean(metadata={'description': allow_file_upload_description})
    allow_ctas: bool = fields.Boolean(metadata={'description': allow_ctas_description})
    allow_cvas: bool = fields.Boolean(metadata={'description': allow_cvas_description})
    allow_dml: bool = fields.Boolean(metadata={'description': allow_dml_description})
    force_ctas_schema: str = fields.String(metadata={'description': force_ctas_schema_description}, allow_none=True, validate=Length(0, 250))
    impersonate_user: bool = fields.Boolean(metadata={'description': impersonate_user_description})
    masked_encrypted_extra: str = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    extra: str = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    server_cert: str = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    sqlalchemy_uri: str = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(1, 1024), sqlalchemy_uri_validator])
    is_managed_externally: bool = fields.Boolean(allow_none=True, dump_default=False)
    external_url: str = fields.String(allow_none=True)
    uuid: str = fields.String(required=False)
    ssh_tunnel: DatabaseSSHTunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)

class DatabasePutSchema(DatabaseParametersSchemaMixin, Schema):

    class Meta:
        unknown = EXCLUDE
    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    database_name: str = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    cache_timeout: int = fields.Integer(metadata={'description': cache_timeout_description}, allow_none=True)
    expose_in_sqllab: bool = fields.Boolean(metadata={'description': expose_in_sqllab_description})
    allow_run_async: bool = fields.Boolean(metadata={'description': allow_run_async_description})
    allow_file_upload: bool = fields.Boolean(metadata={'description': allow_file_upload_description})
    allow_ctas: bool = fields.Boolean(metadata={'description': allow_ctas_description})
    allow_cvas: bool = fields.Boolean(metadata={'description': allow_cvas_description})
    allow_dml: bool = fields.Boolean(metadata={'description': allow_dml_description})
    force_ctas_schema: str = fields.String(metadata={'description': force_ctas_schema_description}, allow_none=True, validate=Length(0, 250))
    impersonate_user: bool = fields.Boolean(metadata={'description': impersonate_user_description})
    masked_encrypted_extra: str = fields.String(metadata={'description': encrypted_extra_description}, allow_none=True, validate=encrypted_extra_validator)
    extra: str = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    server_cert: str = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    sqlalchemy_uri: str = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(0, 1024), sqlalchemy_uri_validator])
    is_managed_externally: bool = fields.Boolean(allow_none=True, dump_default=False)
    external_url: str = fields.String(allow_none=True)
    ssh_tunnel: DatabaseSSHTunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)
    uuid: str = fields.String(required=False)

class DatabaseTestConnectionSchema(DatabaseParametersSchemaMixin, Schema):

    rename_encrypted_extra = pre_load(rename_encrypted_extra)
    database_name: str = fields.String(metadata={'description': database_name_description}, allow_none=True, validate=Length(1, 250))
    impersonate_user: bool = fields.Boolean(metadata={'description': impersonate_user_description})
    extra: str = fields.String(metadata={'description': extra_description}, validate=extra_validator)
    masked_encrypted_extra: str = fields.String(metadata={'description': encrypted_extra_description}, validate=encrypted_extra_validator, allow_none=True)
    server_cert: str = fields.String(metadata={'description': server_cert_description}, allow_none=True, validate=server_cert_validator)
    sqlalchemy_uri: str = fields.String(metadata={'description': sqlalchemy_uri_description}, validate=[Length(1, 1024), sqlalchemy_uri_validator])
    ssh_tunnel: DatabaseSSHTunnel = fields.Nested(DatabaseSSHTunnel, allow_none=True)

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

    deferrable: bool = fields.Bool()
    initially: bool = fields.Bool()
    match: bool = fields.Bool()
    ondelete: bool = fields.Bool()
    onupdate: bool = fields.Bool()

class TableMetadataColumnsResponseSchema(Schema):

    keys: list[str] = fields.List(fields.String(), metadata={'description': ''})
    longType: str = fields.String(metadata={'description': 'The actual backend long type for the column'})
    name: str = fields.String(metadata={'description': 'The column name'})
    type: str = fields.String(metadata={'description': 'The column type'})
    duplicates_constraint: str = fields.String(required=False)

class TableMetadataForeignKeysIndexesResponseSchema(Schema):

    column_names: list[str] = fields.List(fields.String(metadata={'description': 'A list of column names that compose the foreign key or  index'}))
    name: str = fields.String(metadata={'description': 'The name of