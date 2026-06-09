from typing import Any

# === Third-party dependency: dbt.artifacts.resources.base ===
class FileHash(dbtClassMixin):
    ...

# === Third-party dependency: dbt.config ===
# Used symbols: RuntimeConfig

# === Third-party dependency: dbt.contracts.graph.manifest ===
class ManifestStateCheck(dbtClassMixin):
    ...
class Manifest(MacroMethods, dbtClassMixin): ...

# === Third-party dependency: dbt.events.types ===
class UnusedResourceConfigPath(WarnLevel): ...

# === Third-party dependency: dbt.flags ===
def set_from_args(args: Namespace, project_flags) -> Any: ...

# === Third-party dependency: dbt.parser.manifest ===
class ManifestLoader:
    def __init__(self, root_project: RuntimeConfig, all_projects: Mapping[str, RuntimeConfig], macro_hook: Optional[Callable[[Manifest], Any]] = ..., file_diff: Optional[FileDiff] = ...) -> None: ...
    def safe_update_project_parser_files_partially(self, project_parser_files: Dict) -> Dict: ...
    def is_partial_parsable(self, manifest: Manifest) -> Tuple[bool, Optional[str]]: ...
def _warn_for_unused_resource_config_paths(manifest: Manifest, config: RuntimeConfig) -> None: ...

# === Third-party dependency: dbt.parser.read_files ===
class FileDiff(dbtClassMixin):
    ...

# === Third-party dependency: dbt.tracking ===
class User:
    def __init__(self, cookie_dir) -> None: ...

# === Third-party dependency: dbt_common.events.event_manager_client ===
def add_callback_to_manager(callback: TCallback) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark

# === Internal dependency: tests.utils ===
class EventCatcher:
    def catch(self, event: EventMsg) -> Any: ...