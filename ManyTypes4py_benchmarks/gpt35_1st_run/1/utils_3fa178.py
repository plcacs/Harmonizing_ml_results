def get_openapi_security_definitions(flat_dependant: Dependant) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
def _get_openapi_operation_parameters(*, dependant: Dependant, schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[str, Any], separate_input_output_schemas: bool = True) -> List[Dict[str, Any]]:
def get_openapi_operation_request_body(*, body_field: ModelField, schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[str, Any], separate_input_output_schemas: bool = True) -> Optional[Dict[str, Any]]:
def generate_operation_id(*, route: BaseRoute, method: str) -> str:
def generate_operation_summary(*, route: BaseRoute, method: str) -> str:
def get_openapi_operation_metadata(*, route: BaseRoute, method: str, operation_ids: Set[str]) -> Dict[str, Any]:
def get_openapi_path(*, route: BaseRoute, operation_ids: Set[str], schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[str, Any], separate_input_output_schemas: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
def get_fields_from_routes(routes: Sequence[BaseRoute]) -> List[ModelField]:
def get_openapi(*, title: str, version: str, openapi_version: str = '3.1.0', summary: Optional[str] = None, description: Optional[str] = None, routes: Optional[Sequence[BaseRoute]], webhooks: Optional[Sequence[BaseRoute]], tags: Optional[List[Dict[str, Any]]] = None, servers: Optional[List[Dict[str, Any]]] = None, terms_of_service: Optional[str] = None, contact: Optional[Dict[str, str]] = None, license_info: Optional[Dict[str, str]] = None, separate_input_output_schemas: bool = True) -> Dict[str, Any]:
