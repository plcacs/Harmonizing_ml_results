def load_project(project_root: str, version_check: bool, profile: Profile, cli_vars: Optional[Dict[str, Any]] = None) -> Project:
    ...

def load_profile(project_root: str, cli_vars: Dict[str, Any], profile_name_override: Optional[str] = None, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> Profile:
    ...

def _project_quoting_dict(proj: Project, profile: Profile) -> Dict[ComponentName, bool]:
    ...

@dataclass
class RuntimeConfig(Project, Profile, AdapterRequiredConfig):
    dependencies: Optional[Dict[str, Any]] = None
    invoked_at: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))

    def __post_init__(self) -> None:
        ...

    @classmethod
    def get_profile(cls, project_root: str, cli_vars: Dict[str, Any], args: Any) -> Profile:
        ...

    @classmethod
    def from_parts(cls, project: Project, profile: Profile, args: Any, dependencies: Optional[Dict[str, Any]] = None) -> RuntimeConfig:
        ...

    def new_project(self, project_root: str) -> RuntimeConfig:
        ...

    def serialize(self) -> Dict[str, Any]:
        ...

    def validate(self) -> None:
        ...

    @classmethod
    def collect_parts(cls, args: Any) -> Tuple[Project, Profile]:
        ...

    @classmethod
    def from_args(cls, args: Any) -> RuntimeConfig:
        ...

    def get_metadata(self) -> ManifestMetadata:
        ...

    def _get_v2_config_paths(self, config: Dict[str, Any], path: Tuple[str, ...], paths: MutableSet[Tuple[str, ...]]) -> frozenset:
        ...

    def _get_config_paths(self, config: Dict[str, Any], path: Tuple[str, ...] = (), paths: Optional[MutableSet[Tuple[str, ...]]] = None) -> frozenset:
        ...

    def get_resource_config_paths(self) -> Dict[str, frozenset]:
        ...

    def warn_for_unused_resource_config_paths(self, resource_fqns: Dict[str, frozenset], disabled: Iterable[Tuple[str, ...]]) -> None:
        ...

    def load_dependencies(self, base_only: bool = False) -> Dict[str, Any]:
        ...

    def clear_dependencies(self) -> None:
        ...

    def load_projects(self, paths: Iterable[Path]) -> Iterator[Tuple[str, Project]]:
        ...

    def _get_project_directories(self) -> Iterable[Path]:
        ...

class UnsetCredentials(Credentials):
    ...

class UnsetProfile(Profile):
    ...

def _is_config_used(path: Tuple[str, ...], fqns: frozenset) -> bool:
    ...
