import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type
import dotenv
import toml
from pydantic import AliasChoices
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, DotEnvSettingsSource, EnvSettingsSource, PydanticBaseSettingsSource
from pydantic_settings.sources import ENV_FILE_SENTINEL, ConfigFileSourceMixin, DotenvType
from prefect.settings.constants import DEFAULT_PREFECT_HOME, DEFAULT_PROFILES_PATH
from prefect.utilities.collections import get_from_dict

class EnvFilterSettingsSource(EnvSettingsSource):
    def __init__(self, settings_cls: Type[BaseSettings], case_sensitive: Optional[bool] = None, env_prefix: Optional[str] = None, env_nested_delimiter: Optional[str] = None, env_ignore_empty: Optional[bool] = None, env_parse_none_str: Optional[bool] = None, env_parse_enums: Optional[bool] = None, env_filter: Optional[List[str]] = None):
        ...

class FilteredDotEnvSettingsSource(DotEnvSettingsSource):
    def __init__(self, settings_cls: Type[BaseSettings], env_file: str = ENV_FILE_SENTINEL, env_file_encoding: Optional[str] = None, case_sensitive: Optional[bool] = None, env_prefix: Optional[str] = None, env_nested_delimiter: Optional[str] = None, env_ignore_empty: Optional[bool] = None, env_parse_none_str: Optional[bool] = None, env_parse_enums: Optional[bool] = None, env_blacklist: Optional[List[str]] = None):
        ...

class ProfileSettingsTomlLoader(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: Type[BaseSettings]):
        ...

class TomlConfigSettingsSourceBase(PydanticBaseSettingsSource, ConfigFileSourceMixin):
    def __init__(self, settings_cls: Type[BaseSettings]):
        ...

class PrefectTomlConfigSettingsSource(TomlConfigSettingsSourceBase):
    def __init__(self, settings_cls: Type[BaseSettings]):
        ...

class PyprojectTomlConfigSettingsSource(TomlConfigSettingsSourceBase):
    def __init__(self, settings_cls: Type[BaseSettings]):
        ...
