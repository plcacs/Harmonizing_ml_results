import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from dbt.adapters.contracts.connection import Credentials, HasCredentials
from dbt.clients.yaml_helper import load_yaml_text
from dbt.contracts.project import ProfileConfig
from dbt.events.types import MissingProfileTarget
from dbt.exceptions import CompilationError, DbtProfileError, DbtProjectError, DbtRuntimeError, ProfileConfigError
from dbt.flags import get_flags
from dbt_common.clients.system import load_file_contents
from dbt_common.dataclass_schema import ValidationError
from dbt_common.events.functions import fire_event
from .renderer import ProfileRenderer

DEFAULT_THREADS: int = 1
INVALID_PROFILE_MESSAGE: str = '\ndbt encountered an error while trying to read your profiles.yml file.\n\n{error_string}\n'

def read_profile(profiles_dir: str) -> Dict[str, Any]:
    path: str = os.path.join(profiles_dir, 'profiles.yml')
    contents: Optional[str] = None
    if os.path.isfile(path):
        try:
            contents = load_file_contents(path, strip=False)
            yaml_content = load_yaml_text(contents)
            if not yaml_content:
                msg: str = f'The profiles.yml file at {path} is empty'
                raise DbtProfileError(INVALID_PROFILE_MESSAGE.format(error_string=msg))
            return yaml_content
        except DbtValidationError as e:
            msg: str = INVALID_PROFILE_MESSAGE.format(error_string=e)
            raise DbtValidationError(msg) from e
    return {}

@dataclass(init=False)
class Profile(HasCredentials):
    profile_name: str
    target_name: str
    threads: int
    credentials: Credentials
    profile_env_vars: Dict[str, Any] = {}
    log_cache_events: bool = get_flags().LOG_CACHE_EVENTS

    def to_profile_info(self, serialize_credentials: bool = False) -> Dict[str, Any]:
        ...

    def to_target_dict(self) -> Dict[str, Any]:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def validate(self) -> None:
        ...

    @staticmethod
    def _credentials_from_profile(profile: Dict[str, Any], profile_name: str, target_name: str) -> Credentials:
        ...

    @staticmethod
    def pick_profile_name(args_profile_name: Optional[str], project_profile_name: Optional[str] = None) -> str:
        ...

    @staticmethod
    def _get_profile_data(profile: Dict[str, Any], profile_name: str, target_name: str) -> Dict[str, Any]:
        ...

    @classmethod
    def from_credentials(cls, credentials: Credentials, threads: int, profile_name: str, target_name: str) -> 'Profile':
        ...

    @classmethod
    def render_profile(cls, raw_profile: Dict[str, Any], profile_name: str, target_override: Optional[str], renderer: ProfileRenderer) -> Tuple[str, Dict[str, Any]]:
        ...

    @classmethod
    def from_raw_profile_info(cls, raw_profile: Dict[str, Any], profile_name: str, renderer: ProfileRenderer, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> 'Profile':
        ...

    @classmethod
    def from_raw_profiles(cls, raw_profiles: Dict[str, Any], profile_name: str, renderer: ProfileRenderer, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> 'Profile':
        ...

    @classmethod
    def render(cls, renderer: ProfileRenderer, project_profile_name: Optional[str], profile_name_override: Optional[str] = None, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> 'Profile':
        ...
