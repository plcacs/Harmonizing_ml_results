env_file_sentinel: str
SettingsSourceCallable: Type[Callable[['BaseSettings'], Dict[str, Any]]
DotenvType: Type[Union[StrPath, List[StrPath], Tuple[StrPath, ...]]

class SettingsError(ValueError):
    pass

class BaseSettings(BaseModel):
    def __init__(__pydantic_self__, _env_file: str = env_file_sentinel, _env_file_encoding: Optional[str] = None, _env_nested_delimiter: Optional[str] = None, _secrets_dir: Optional[str] = None, **values: Any) -> None:
        ...

    def _build_values(self, init_kwargs: Dict[str, Any], _env_file: Optional[str] = None, _env_file_encoding: Optional[str] = None, _env_nested_delimiter: Optional[str] = None, _secrets_dir: Optional[str] = None) -> Dict[str, Any]:
        ...

    class Config(BaseConfig):
        env_prefix: str = ''
        env_file: Optional[str] = None
        env_file_encoding: Optional[str] = None
        env_nested_delimiter: Optional[str] = None
        secrets_dir: Optional[str] = None
        validate_all: bool = True
        extra: Extra = Extra.forbid
        arbitrary_types_allowed: bool = True
        case_sensitive: bool = False

        @classmethod
        def prepare_field(cls, field: ModelField) -> None:
            ...

        @classmethod
        def customise_sources(cls, init_settings: InitSettingsSource, env_settings: EnvSettingsSource, file_secret_settings: SecretsSettingsSource) -> Tuple[InitSettingsSource, EnvSettingsSource, SecretsSettingsSource]:
            ...

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            ...

class InitSettingsSource:
    def __init__(self, init_kwargs: Dict[str, Any]) -> None:
        ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        ...

class EnvSettingsSource:
    def __init__(self, env_file: Union[StrPath, List[StrPath], Tuple[StrPath, ...], env_file_encoding: Optional[str], env_nested_delimiter: Optional[str], env_prefix_len: int) -> None:
        ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        ...

    def _read_env_files(self, case_sensitive: bool) -> Dict[str, str]:
        ...

    def field_is_complex(self, field: ModelField) -> Tuple[bool, bool]:
        ...

    def explode_env_vars(self, field: ModelField, env_vars: Dict[str, str]) -> Dict[str, Any]:
        ...

class SecretsSettingsSource:
    def __init__(self, secrets_dir: Optional[str]) -> None:
        ...

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        ...

def read_env_file(file_path: Union[str, os.PathLike], *, encoding: Optional[str] = None, case_sensitive: bool = False) -> Dict[str, str]:
    ...

def find_case_path(dir_path: Union[str, os.PathLike], file_name: str, case_sensitive: bool) -> Optional[Union[str, os.PathLike]]:
    ...
