from typing import TypedDict
from marshmallow import Schema, fields
from superset.db_engine_specs.base import BaseEngineSpec
from superset.models.core import Database
from superset.models.sql_lab import Query
from superset.errors import SupersetError, ErrorLevel, SupersetErrorType

class SnowflakeParametersType(TypedDict):
    pass

class SnowflakeParametersSchema(Schema):
    username: fields.Str(required=True)
    password: fields.Str(required=True)
    account: fields.Str(required=True)
    database: fields.Str(required=True)
    role: fields.Str(required=True)
    warehouse: fields.Str(required=True)

class SnowflakeEngineSpec(BaseEngineSpec):
    def get_extra_params(database: Database) -> dict:
        ...

    def adjust_engine_params(cls, uri: URL, connect_args: dict, catalog: str = None, schema: str = None) -> tuple:
        ...

    def get_schema_from_engine_params(cls, sqlalchemy_uri: str, connect_args: dict) -> str:
        ...

    def get_default_catalog(cls, database: Database) -> str:
        ...

    def get_catalog_names(cls, database: Database, inspector: Inspector) -> set:
        ...

    def epoch_to_dttm(cls) -> str:
        ...

    def epoch_ms_to_dttm(cls) -> str:
        ...

    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Any = None) -> str:
        ...

    def mutate_db_for_connection_test(database: Database) -> None:
        ...

    def get_cancel_query_id(cls, cursor: Any, query: Query) -> Any:
        ...

    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: Any) -> bool:
        ...

    def build_sqlalchemy_uri(cls, parameters: dict, encrypted_extra: Any = None) -> str:
        ...

    def get_parameters_from_uri(cls, uri: str, encrypted_extra: Any = None) -> dict:
        ...

    def validate_parameters(cls, properties: dict) -> list:
        ...

    def parameters_json_schema(cls) -> dict:
        ...

    def update_params_from_encrypted_extra(database: Database, params: dict) -> None:
        ...
