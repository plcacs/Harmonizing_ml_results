    def __init__(self, scope: Construct, id: str, source_dir: str, stage_config: Optional[Dict[str, Any]] = None, preserve_logical_ids: bool = True, **kwargs: Any) -> None:
    def _filter_resources(self, template: Dict[str, Any], resource_type: str) -> List[Dict[str, Any]]:
    def get_resource(self, resource_name: str) -> Any:
    def get_role(self, role_name: str) -> iam.Role:
    def get_function(self, function_name: str) -> lambda_.Function:
    def add_environment_variable(self, key: str, value: str, function_name: str) -> None:
