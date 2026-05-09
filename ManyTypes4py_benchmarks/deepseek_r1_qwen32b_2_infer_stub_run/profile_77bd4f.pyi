import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union
from dbt.adapters.contracts.connection import Credentials, HasCredentials
from dbt.contracts.project import ProfileConfig
from dbt.events.types import MissingProfileTarget
from dbt.exceptions import (CompilationError, DbtProfileError, DbtProjectError,
                           DbtRuntimeError, ProfileConfigError)
from dbt.flags import get_flags
from dbt_common.clients.system import load_file_contents
from dbt_common.dataclass_schema import ValidationError
from dbt_common.events.functions import fire_event
from dbt_common.exceptions import DbtValidationError
from .renderer import ProfileRenderer

DEFAULT_THREADS: int
INVALID_PROFILE_MESSAGE: str

def read_profile(profiles_dir: str) -> Dict[Any, Any]: ...

@dataclass(init=False)
class Profile(HasCredentials):
    profile_name: str
    target_name: str
    threads: int
    credentials: Credentials
    profile_env_vars: Dict[str, Any]
    log_cache_events: bool

    def __init__(self, profile_name: str, target_name: str, threads: int, credentials: Credentials) -> None: ...

    def to_profile_info(self, serialize_credentials: bool = False) -> Dict[str, Any]: ...

    def to_target_dict(self) -> Dict[str, Any]: ...

    def __eq__(self, other: Any) -> bool: ...

    def validate(self) -> None: ...

    @staticmethod
    def _credentials_from_profile(profile: Dict[str, Any], profile_name: str, target_name: str) -> Credentials: ...

    @staticmethod
    def pick_profile_name(args_profile_name: Optional[str], project_profile_name: Optional[str] = None) -> str: ...

    @staticmethod
    def _get_profile_data(profile: Dict[str, Any], profile_name: str, target_name: str) -> Dict[str, Any]: ...

    @classmethod
    def from_credentials(cls, credentials: Credentials, threads: int, profile_name: str, target_name: str) -> 'Profile': ...

    @classmethod
    def render_profile(cls, raw_profile: Dict[str, Any], profile_name: str, target_override: Optional[str], renderer: ProfileRenderer) -> Tuple[str, Dict[str, Any]]: ...

    @classmethod
    def from_raw_profile_info(cls, raw_profile: Dict[str, Any], profile_name: str, renderer: ProfileRenderer, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> 'Profile': ...

    @classmethod
    def from_raw_profiles(cls, raw_profiles: Dict[str, Any], profile_name: str, renderer: ProfileRenderer, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> 'Profile': ...

    @classmethod
    def render(cls, renderer: ProfileRenderer, project_profile_name: Optional[str], profile_name_override: Optional[str] = None, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> 'Profile': ...