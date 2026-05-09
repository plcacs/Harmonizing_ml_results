import re
from urllib.parse import urljoin
import typesystem
from apistar.document import Document, Field, Link, Section
from apistar.schemas.jsonschema import JSON_SCHEMA
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

SCHEMA_REF = typesystem.Object(properties={'$ref': typesystem.String(pattern='^#/components/schemas/')})
REQUESTBODY_REF = typesystem.Object(properties={'$ref': typesystem.String(pattern='^#/components/requestBodies/')})
RESPONSE_REF = typesystem.Object(properties={'$ref': typesystem.String(pattern='^#/components/responses/')})

OpenAPI = ...

def load(self, data: Dict[str, Any]) -> Document:
    ...

def get_schema_definitions(self, data: Dict[str, Any]) -> typesystem.SchemaDefinitions:
    ...

def get_content(self, data: Dict[str, Any], base_url: str, schema_definitions: typesystem.SchemaDefinitions) -> Tuple[List[Link], List[Section]]:
    ...

def get_link(self, base_url: str, path: str, path_info: Dict[str, Any], operation: str, operation_info: Dict[str, Any], schema_definitions: typesystem.SchemaDefinitions) -> Optional[Link]:
    ...

def get_field(self, parameter: Dict[str, Any], schema_definitions: typesystem.SchemaDefinitions) -> Field:
    ...
