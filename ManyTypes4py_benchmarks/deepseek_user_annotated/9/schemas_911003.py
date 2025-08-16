from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union
from pathlib import Path
from werkzeug.datastructures import FileStorage
from marshmallow import Schema, fields, ValidationError, EXCLUDE
from marshmallow.validate import Length, OneOf, Range
from flask_babel import lazy_gettext as _
from sqlalchemy import MetaData
from superset.models.core import ConfigurationMethod, Database
from superset.commands.database.uploaders.base import UploadFileType
from superset.constants import PASSWORD_MASK
from superset.databases.types import EncryptedDict, EncryptedField, EncryptedString
from superset.utils import json
from superset.utils.core import markdown, parse_ssl_cert

# Type aliases
JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class DatabaseParametersSchemaMixin:
    engine: fields.String = fields.String(
        allow_none=True, metadata={"description": "SQLAlchemy engine to use"}
    )
    driver: fields.String = fields.String(
        allow_none=True, metadata={"description": "SQLAlchemy driver to use"}
    )
    parameters: fields.Dict = fields.Dict(
        keys=fields.String(),
        values=fields.Raw(),
        metadata={"description": "DB-specific parameters for configuration"},
    )
    configuration_method: fields.Enum = fields.Enum(
        ConfigurationMethod,
        by_value=True,
        metadata={"description": configuration_method_description},
        load_default=ConfigurationMethod.SQLALCHEMY_FORM,
    )

class DatabaseSSHTunnel(Schema):
    id: fields.Integer = fields.Integer(
        allow_none=True, metadata={"description": "SSH Tunnel ID (for updates)"}
    )
    server_address: fields.String = fields.String()
    server_port: fields.Integer = fields.Integer()
    username: fields.String = fields.String()
    password: fields.String = fields.String(required=False)
    private_key: fields.String = fields.String(required=False)
    private_key_password: fields.String = fields.String(required=False)

class TableMetadataOptionsResponse(TypedDict):
    deferrable: bool
    initially: bool
    match: bool
    ondelete: bool
    onupdate: bool

class TableMetadataColumnsResponse(TypedDict, total=False):
    keys: List[str]
    longType: str
    name: str
    type: str
    duplicates_constraint: Optional[str]
    comment: Optional[str]

class TableMetadataForeignKeysIndexesResponse(TypedDict):
    column_names: List[str]
    name: str
    options: TableMetadataOptionsResponse
    referred_columns: List[str]
    referred_schema: str
    referred_table: str
    type: str

class TableMetadataPrimaryKeyResponse(TypedDict):
    column_names: List[str]
    name: str
    type: str

class TableMetadataResponse(TypedDict):
    name: str
    columns: List[TableMetadataColumnsResponse]
    foreignKeys: List[TableMetadataForeignKeysIndexesResponse]
    indexes: List[TableMetadataForeignKeysIndexesResponse]
    primaryKey: TableMetadataPrimaryKeyResponse
    selectStar: str
    comment: Optional[str]

class TableMetadataOptionsResponseSchema(Schema):
    deferrable: fields.Bool = fields.Bool()
    initially: fields.Bool = fields.Bool()
    match: fields.Bool = fields.Bool()
    ondelete: fields.Bool = fields.Bool()
    onupdate: fields.Bool = fields.Bool()

class TableMetadataColumnsResponseSchema(Schema):
    keys: fields.List = fields.List(fields.String(), metadata={"description": ""})
    longType: fields.String = fields.String(
        metadata={"description": "The actual backend long type for the column"}
    )
    name: fields.String = fields.String(metadata={"description": "The column name"})
    type: fields.String = fields.String(metadata={"description": "The column type"})
    duplicates_constraint: fields.String = fields.String(required=False)

class TableMetadataForeignKeysIndexesResponseSchema(Schema):
    column_names: fields.List = fields.List(
        fields.String(
            metadata={
                "description": "A list of column names that compose the foreign key or index"
            }
        )
    )
    name: fields.String = fields.String(
        metadata={"description": "The name of the foreign key or index"}
    )
    options: fields.Nested = fields.Nested(TableMetadataOptionsResponseSchema)
    referred_columns: fields.List = fields.List(fields.String())
    referred_schema: fields.String = fields.String()
    referred_table: fields.String = fields.String()
    type: fields.String = fields.String()

class TableMetadataPrimaryKeyResponseSchema(Schema):
    column_names: fields.List = fields.List(
        fields.String(
            metadata={
                "description": "A list of column names that compose the primary key"
            }
        )
    )
    name: fields.String = fields.String(metadata={"description": "The primary key index name"})
    type: fields.String = fields.String()

class TableMetadataResponseSchema(Schema):
    name: fields.String = fields.String(metadata={"description": "The name of the table"})
    columns: fields.List = fields.List(
        fields.Nested(TableMetadataColumnsResponseSchema),
        metadata={"description": "A list of columns and their metadata"},
    )
    foreignKeys: fields.List = fields.List(
        fields.Nested(TableMetadataForeignKeysIndexesResponseSchema),
        metadata={"description": "A list of foreign keys and their metadata"},
    )
    indexes: fields.List = fields.List(
        fields.Nested(TableMetadataForeignKeysIndexesResponseSchema),
        metadata={"description": "A list of indexes and their metadata"},
    )
    primaryKey: fields.Nested = fields.Nested(
        TableMetadataPrimaryKeyResponseSchema,
        metadata={"description": "Primary keys metadata"},
    )
    selectStar: fields.String = fields.String(metadata={"description": "SQL select star"})

class TableExtraMetadataResponseSchema(Schema):
    metadata: fields.Dict = fields.Dict()
    partitions: fields.Dict = fields.Dict()
    clustering: fields.Dict = fields.Dict()

class SelectStarResponseSchema(Schema):
    result: fields.String = fields.String(metadata={"description": "SQL select star"})

class SchemasResponseSchema(Schema):
    result: fields.List = fields.List(
        fields.String(metadata={"description": "A database schema name"})
    )

class CatalogsResponseSchema(Schema):
    result: fields.List = fields.List(
        fields.String(metadata={"description": "A database catalog name"})
    )

class DatabaseTablesResponse(Schema):
    extra: fields.Dict = fields.Dict(
        metadata={"description": "Extra data used to specify column metadata"}
    )
    type: fields.String = fields.String(metadata={"description": "table or view"})
    value: fields.String = fields.String(metadata={"description": "The table or view name"})

class ValidateSQLRequest(Schema):
    sql: fields.String = fields.String(
        required=True, metadata={"description": "SQL statement to validate"}
    )
    catalog: fields.String = fields.String(required=False, allow_none=True)
    schema: fields.String = fields.String(required=False, allow_none=True)
    template_params: fields.Dict = fields.Dict(required=False, allow_none=True)

class ValidateSQLResponse(Schema):
    line_number: fields.Integer = fields.Integer()
    start_column: fields.Integer = fields.Integer()
    end_column: fields.Integer = fields.Integer()
    message: fields.String = fields.String()

class DatabaseRelatedChart(Schema):
    id: fields.Integer = fields.Integer()
    slice_name: fields.String = fields.String()
    viz_type: fields.String = fields.String()

class DatabaseRelatedDashboard(Schema):
    id: fields.Integer = fields.Integer()
    json_metadata: fields.Dict = fields.Dict()
    slug: fields.String = fields.String()
    title: fields.String = fields.String()

class DatabaseRelatedCharts(Schema):
    count: fields.Integer = fields.Integer(metadata={"description": "Chart count"})
    result: fields.List = fields.List(
        fields.Nested(DatabaseRelatedChart),
        metadata={"description": "A list of dashboards"},
    )

class DatabaseRelatedDashboards(Schema):
    count: fields.Integer = fields.Integer(metadata={"description": "Dashboard count"})
    result: fields.List = fields.List(
        fields.Nested(DatabaseRelatedDashboard),
        metadata={"description": "A list of dashboards"},
    )

class DatabaseRelatedObjectsResponse(Schema):
    charts: fields.Nested = fields.Nested(DatabaseRelatedCharts)
    dashboards: fields.Nested = fields.Nested(DatabaseRelatedDashboards)

class DatabaseFunctionNamesResponse(Schema):
    function_names: fields.List = fields.List(fields.String())

class ImportV1DatabaseExtraSchema(Schema):
    metadata_params: fields.Dict = fields.Dict(keys=fields.Str(), values=fields.Raw())
    engine_params: fields.Dict = fields.Dict(keys=fields.Str(), values=fields.Raw())
    metadata_cache_timeout: fields.Dict = fields.Dict(keys=fields.Str(), values=fields.Integer())
    schemas_allowed_for_csv_upload: fields.List = fields.List(fields.String())
    cost_estimate_enabled: fields.Boolean = fields.Boolean()
    allows_virtual_table_explore: fields.Boolean = fields.Boolean(required=False)
    cancel_query_on_windows_unload: fields.Boolean = fields.Boolean(required=False)
    disable_data_preview: fields.Boolean = fields.Boolean(required=False)
    disable_drill_to_detail: fields.Boolean = fields.Boolean(required=False)
    allow_multi_catalog: fields.Boolean = fields.Boolean(required=False)
    version: fields.String = fields.String(required=False, allow_none=True)

class ImportV1DatabaseSchema(Schema):
    database_name: fields.String = fields.String(required=True)
    sqlalchemy_uri: fields.String = fields.String(required=True)
    password: fields.String = fields.String(allow_none=True)
    cache_timeout: fields.Integer = fields.Integer(allow_none=True)
    expose_in_sqllab: fields.Boolean = fields.Boolean()
    allow_run_async: fields.Boolean = fields.Boolean()
    allow_ctas: fields.Boolean = fields.Boolean()
    allow_cvas: fields.Boolean = fields.Boolean()
    allow_dml: fields.Boolean = fields.Boolean(required=False)
    allow_csv_upload: fields.Boolean = fields.Boolean()
    impersonate_user: fields.Boolean = fields.Boolean()
    extra: fields.Nested = fields.Nested(ImportV1DatabaseExtraSchema)
    uuid: fields.UUID = fields.UUID(required=True)
    version: fields.String = fields.String(required=True)
    is_managed_externally: fields.Boolean = fields.Boolean(allow_none=True, dump_default=False)
    external_url: fields.String = fields.String(allow_none=True)
    ssh_tunnel: fields.Nested = fields.Nested(DatabaseSSHTunnel, allow_none=True)

class DatabaseSchemaAccessForFileUploadResponse(Schema):
    schemas: fields.List = fields.List(
        fields.String(),
        metadata={
            "description": "The list of schemas allowed for the database to upload information"
        },
    )

class EngineInformationSchema(Schema):
    supports_file_upload: fields.Boolean = fields.Boolean(
        metadata={"description": "Users can upload files to the database"}
    )
    disable_ssh_tunneling: fields.Boolean = fields.Boolean(
        metadata={"description": "SSH tunnel is not available to the database"}
    )
    supports_dynamic_catalog: fields.Boolean = fields.Boolean(
        metadata={
            "description": "The database supports multiple catalogs in a single connection"
        }
    )
    supports_oauth2: fields.Boolean = fields.Boolean(
        metadata={"description": "The database supports OAuth2"}
    )

class DatabaseConnectionSchema(Schema):
    allow_ctas: fields.Boolean = fields.Boolean(metadata={"description": allow_ctas_description})
    allow_cvas: fields.Boolean = fields.Boolean(metadata={"description": allow_cvas_description})
    allow_dml: fields.Boolean = fields.Boolean(metadata={"description": allow_dml_description})
    allow_file_upload: fields.Boolean = fields.Boolean(
        metadata={"description": allow_file_upload_description}
    )
    allow_run_async: fields.Boolean = fields.Boolean(
        metadata={"description": allow_run_async_description}
    )
    backend: fields.String = fields.String(
        allow_none=True, metadata={"description": "SQLAlchemy engine to use"}
    )
    cache_timeout: fields.Integer = fields.Integer(
        metadata={"description": cache_timeout_description}, allow_none=True
    )
    configuration_method: fields.String = fields.String(
        metadata={"description": configuration_method_description},
    )
    database_name: fields.String = fields.String(
        metadata={"description": database_name_description},
        allow_none=True,
        validate=Length(1, 250),
    )
    driver: fields.String = fields.String(
        allow_none=True, metadata={"description": "SQLAlchemy driver to use"}
    )
    engine_information: fields.Nested = fields.Nested(EngineInformationSchema)
    expose_in_sqllab: fields.Boolean = fields.Boolean(
        metadata={"description": expose_in_sqllab_description}
    )
    extra: fields.String = fields.String(
        metadata={"description": extra_description}, validate=extra_validator
    )
    force_ctas_schema: fields.String = fields.String(
        metadata={"description": force_ctas_schema_description},
        allow_none=True,
        validate=Length(0, 250),
    )
    id: fields.Integer = fields.Integer(metadata={"description": "Database ID (for updates)"})
    impersonate_user: fields.Boolean = fields.Boolean(
        metadata={"description": impersonate_user_description}
    )
    is_managed_externally: fields.Boolean = fields.Boolean(allow_none=True, dump_default=False)
    server_cert: fields.String = fields.String(
        metadata={"description": server_cert_description},
        allow_none=True,
        validate=server_cert_validator,
    )
    uuid: fields.String = fields.String(required=False)
    ssh_tunnel: fields.Nested = fields.Nested(DatabaseSSHTunnel, allow_none=True)
    masked_encrypted_extra: fields.String = fields.String(
        metadata={"description": encrypted_extra_description},
        validate=encrypted_extra_validator,
        allow_none=True,
    )
    parameters: fields.Dict = fields.Dict(
        keys=fields.String(),
        values=fields.Raw(),
        metadata={"description": "DB-specific parameters for configuration"},
    )
    parameters_schema: fields.Dict = fields.Dict(
        keys=fields.String(),
        values=fields.Raw(),
        metadata={
            "description": (
                "JSONSchema for configuring the database by "
                "parameters instead of SQLAlchemy URI"
            ),
        },
    )
    sqlalchemy_uri: fields.String = fields.String(
        metadata={"description": sqlalchemy_uri_description},
        validate=[Length(1, 1024), sqlalchemy_uri_validator],
    )

class DelimitedListField(fields.List):
    pass

class BaseUploadFilePostSchemaMixin(Schema):
    pass

class UploadPostSchema(BaseUploadFilePostSchemaMixin):
    type: fields.Enum = fields.Enum(
        UploadFileType,
        required=True,
        by_value=True,
        metadata={"description": "File type to upload"},
    )
    already_exists: fields.String = fields.String(
        load_default="fail",
        validate=OneOf(choices=("fail", "replace", "append")),
        metadata={
            "description": "What to do if the table already exists accepts: fail, replace, append"
        },
    )
    index_label: fields.String = fields.String(
        metadata={"description": "Index label for index column."}
    )
    columns_read: DelimitedListField = DelimitedListField(
        fields.String(),
        metadata={"description": "A List of the column names that should be read"},
    )
    dataframe_index: fields.Boolean = fields.Boolean(
        metadata={"description": "Write dataframe index as a column."}
    )
    schema: fields.String = fields.String(
        metadata={"description": "The schema to upload the data file to."}
    )
    table_name: fields.String = fields.String(
        required=True,
        validate=[Length(min=1, max=10000)],
        allow_none=False,
        metadata={"description": "The name of the table to be created/appended"},
    )
    file: fields.Raw = fields.Raw(
        required=True,
        metadata={
            "description": "The file to upload",
            "type": "string",
            "format": "text/csv",
        },
    )
    delimiter: fields.String = fields.String(
        metadata={
            "description": "[CSV only] The character used to separate values in the CSV file"
        }
    )
    column_data_types: fields.String = fields.String(
        metadata={
            "description": "[CSV only] A dictionary with column names and their data types"
        }
    )
    day_first: fields.Boolean = fields.Boolean(
        metadata={
            "description": "[CSV only] DD/MM format dates, international and European format"
        }
    )
    skip_blank_lines: fields.Boolean = fields.Boolean(
        metadata={"description": "[CSV only] Skip blank lines in the CSV file."}
    )
    skip_initial_space: fields.Boolean = fields.Boolean(
        metadata={"description": "[CSV only] Skip spaces after delimiter."}
    )
    column_dates: DelimitedListField = DelimitedListField(
        fields.String(),
        metadata={
            "description": "[CSV and Excel only] A list of column names that should be parsed as dates"
        },
    )
    decimal_character: fields.String = fields.String(
        metadata={
            "description": "[CSV and Excel only] Character to recognize as decimal point"
        }
    )
    header_row: fields.Integer = fields.Integer(
        metadata={
            "description": "[CSV and Excel only] Row containing the headers to use as column names"
        }
    )
    index_column: fields.String = fields.String(
        metadata={
            "description": "[CSV and Excel only] Column to use as the row labels of the dataframe"
        }
    )
    null_values: DelimitedListField = DelimitedListField(
        fields.String(),
        metadata={
            "description": "[CSV and Excel only] A list of strings that should be treated as null"
        },
    )
    rows_to_read: fields.Integer = fields.Integer(
        metadata={
            "description": "[CSV and Excel only] Number of rows to read from the file"
        },
        allow_none=True,
        validate=Range(min=1),
    )
   