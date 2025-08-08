from typing import Dict, Any, List, Tuple, Callable, ContextManager

def _write_profiles_yml(profiles_dir: str, dbt_profile_contents: Dict[str, Any]) -> None:
    ...

def environ(env: Dict[str, str]) -> ContextManager[None]:
    ...

class TestProfilesMayNotExist:
    def test_debug(self, project: Any) -> None:
        ...

    def test_deps(self, project: Any) -> None:
        ...

class TestProfiles:
    def dbt_debug(self, project_dir_cli_arg: str = None, profiles_dir_cli_arg: str = None) -> Tuple[List[str], str]:
        ...

    def test_profiles(self, project_dir_cli_arg: str, working_directory: str, write_profiles_yml: Callable, dbt_profile_data: Dict[str, Any], profiles_home_root: str, profiles_project_root: str, profiles_flag_root: str, profiles_env_root: str, request: Any) -> None:
        ...
