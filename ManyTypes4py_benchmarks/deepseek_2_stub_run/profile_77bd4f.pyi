```python
import os
from typing import Any, Dict, Optional, Tuple
from dbt.adapters.contracts.connection import Credentials, HasCredentials
from dbt.contracts.project import ProfileConfig
from dbt.exceptions import CompilationError, DbtProfileError, DbtProjectError, DbtRuntimeError, ProfileConfigError
from dbt_common.dataclass_schema import ValidationError
from .renderer import ProfileRenderer

DEFAULT_THREADS: int = ...

def read_profile(profiles_dir: Any) -> Dict[Any, Any]: ...

class Profile(HasCredentials):
    profile_name: Any
    target_name: Any
    threads: Any
    credentials: Any
    profile_env_vars: Dict[Any, Any]
    log_cache_events: bool
    
    def __init__(self, profile_name: Any, target_name: Any, threads: Any, credentials: Any) -> None: ...
    
    def to_profile_info(self, serialize_credentials: bool = ...) -> Dict[str, Any]: ...
    
    def to_target_dict(self) -> Dict[str, Any]: ...
    
    def __eq__(self, other: Any) -> Any: ...
    
    def validate(self) -> None: ...
    
    @staticmethod
    def _credentials_from_profile(profile: Dict[str, Any], profile_name: str, target_name: str) -> Credentials: ...
    
    @staticmethod
    def pick_profile_name(args_profile_name: Optional[str], project_profile_name: Optional[str] = ...) -> str: ...
    
    @staticmethod
    def _get_profile_data(profile: Dict[str, Any], profile_name: str, target_name: str) -> Dict[str, Any]: ...
    
    @classmethod
    def from_credentials(cls, credentials: Credentials, threads: Any, profile_name: str, target_name: str) -> 'Profile': ...
    
    @classmethod
    def render_profile(cls, raw_profile: Dict[str, Any], profile_name: str, target_override: Optional[str], renderer: ProfileRenderer) -> Tuple[str, Dict[str, Any]]: ...
    
    @classmethod
    def from_raw_profile_info(cls, raw_profile: Dict[str, Any], profile_name: str, renderer: ProfileRenderer, target_override: Optional[str] = ..., threads_override: Optional[int] = ...) -> 'Profile': ...
    
    @classmethod
    def from_raw_profiles(cls, raw_profiles: Dict[str, Any], profile_name: str, renderer: ProfileRenderer, target_override: Optional[str] = ..., threads_override: Optional[int] = ...) -> 'Profile': ...
    
    @classmethod
    def render(cls, renderer: ProfileRenderer, project_profile_name: Optional[str], profile_name_override: Optional[str] = ..., target_override: Optional[str] = ..., threads_override: Optional[int] = ...) -> 'Profile': ...
```