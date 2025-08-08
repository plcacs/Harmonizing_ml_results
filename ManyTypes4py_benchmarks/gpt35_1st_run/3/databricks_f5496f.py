from __future__ import annotations
from datetime import datetime
from typing import Any, TYPE_CHECKING, TypedDict, Union
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
    access_token: fields.Str = fields.Str(required=True)
    host: fields.Str = fields.Str(required=True)
    port: fields.Integer = fields.Integer(required=True, metadata={'description': __('Database port')}, validate=Range(min=0, max=2 ** 16, max_inclusive=False)
    encryption: fields.Boolean = fields.Boolean(required=False, metadata={'description': __('Use an encrypted connection to the database')})

class DatabricksBaseParametersType(TypedDict):
    pass

class DatabricksNativeSchema(DatabricksBaseSchema):
    database: fields.Str = fields.Str(required=True)

class DatabricksNativePropertiesSchema(DatabricksNativeSchema):
    http_path: fields.Str = fields.Str(required=True)

class DatabricksNativeParametersType(DatabricksBaseParametersType):
    pass

class DatabricksNativePropertiesType(TypedDict):
    pass

class DatabricksPythonConnectorSchema(DatabricksBaseSchema):
    http_path_field: fields.Str = fields.Str(required=True)
    default_catalog: fields.Str = fields.Str(required=True)
    default_schema: fields.Str = fields.Str(required=True)

class DatabricksPythonConnectorParametersType(DatabricksBaseParametersType):
    pass

class DatabricksPythonConnectorPropertiesType(TypedDict):
    pass
