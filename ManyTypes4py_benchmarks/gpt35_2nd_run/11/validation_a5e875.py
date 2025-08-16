def is_valid_schema(schema: Schema, preprocess: bool = True) -> None:
def validate(obj: Any, schema: Schema, raise_on_error: bool = False, preprocess: bool = True, ignore_required: bool = False, allow_none_with_default: bool = False) -> List[JSONSchemaValidationError]:
def is_valid(obj: Any, schema: Schema) -> bool:
def prioritize_placeholder_errors(errors: List[JSONSchemaValidationError]) -> List[JSONSchemaValidationError]:
def build_error_obj(errors: List[JSONSchemaValidationError]) -> Dict[str, Any]:
def _fix_null_typing(key: str, schema: Schema, required_fields: List[str], allow_none_with_default: bool = False) -> None:
def _fix_tuple_items(schema: Schema) -> None:
def process_properties(properties: Dict[str, Schema], required_fields: List[str], allow_none_with_default: bool = False) -> None:
def preprocess_schema(schema: Schema, allow_none_with_default: bool = False) -> Schema:
