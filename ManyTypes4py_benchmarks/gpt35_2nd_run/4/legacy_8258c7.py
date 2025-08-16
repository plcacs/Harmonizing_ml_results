    def __init__(self, name: str, default: Any, type_: Type, accessor: Optional[str] = None) -> None:

    def is_secret(self) -> bool:

    def default(self) -> Any:

    def value(self) -> Any:

    def value_from(self, settings: Any) -> Any:

    def __bool__(self) -> bool:

    def __str__(self) -> str:

    def __repr__(self) -> str:

    def __eq__(self, __o: Any) -> bool:

    def __hash__(self) -> int:

def _env_var_to_accessor(env_var: str) -> str:

@cache
def _get_valid_setting_names(cls: Type) -> Set[str]:

@cache
def _get_settings_fields(settings: Any, accessor_prefix: Optional[str] = None) -> Dict[str, Setting]:
