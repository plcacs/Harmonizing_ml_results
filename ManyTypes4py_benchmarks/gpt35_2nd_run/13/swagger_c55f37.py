from typing import Any, Dict, List, Optional

class Swagger:
    def load(self, data: Dict[str, Any]) -> Document:
        ...

    def get_schema_definitions(self, data: Dict[str, Any]) -> typesystem.SchemaDefinitions:
        ...

    def get_content(self, data: Dict[str, Any], base_url: str, schema_definitions: typesystem.SchemaDefinitions) -> List[Union[Link, Section]]:
        ...

    def get_link(self, base_url: str, path: str, path_info: Dict[str, Any], operation: str, operation_info: Dict[str, Any], schema_definitions: typesystem.SchemaDefinitions) -> Optional[Link]:
        ...

    def get_field(self, parameter: Dict[str, Any], schema_definitions: typesystem.SchemaDefinitions) -> Field:
        ...
