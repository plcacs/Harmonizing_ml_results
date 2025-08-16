    def __init__(self, path: Path) -> None:
    def __enter__(self) -> 'PathModifier':
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    @classmethod
    def build_search_paths(cls, config: Config, user_subdir: str = None, extra_dirs: list[str] = None) -> list[Path]:
    @classmethod
    def _get_valid_object(cls, module_path: Path, object_name: str, enum_failed: bool = False) -> Iterator[tuple[Any, str]]:
    @classmethod
    def _search_object(cls, directory: Path, *, object_name: str, add_source: bool = False) -> tuple[Any, Path]:
    @classmethod
    def _load_object(cls, paths: list[Path], *, object_name: str, add_source: bool = False, kwargs: dict) -> Any:
    @classmethod
    def load_object(cls, object_name: str, config: Config, *, kwargs: dict, extra_dir: str = None) -> Any:
    @classmethod
    def search_all_objects(cls, config: Config, enum_failed: bool, recursive: bool = False) -> list[dict[str, Any]]:
    @classmethod
    def _build_rel_location(cls, directory: Path, entry: Path) -> str:
    @classmethod
    def _search_all_objects(cls, directory: Path, enum_failed: bool, recursive: bool = False, basedir: Path = None) -> list[dict[str, Any]]:
