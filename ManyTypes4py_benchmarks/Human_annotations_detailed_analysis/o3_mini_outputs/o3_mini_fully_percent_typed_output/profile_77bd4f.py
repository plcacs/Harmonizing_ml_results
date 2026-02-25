import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from dbt.adapters.contracts.connection import Credentials, HasCredentials
from dbt.clients.yaml_helper import load_yaml_text
from dbt.contracts.project import ProfileConfig
from dbt.events.types import MissingProfileTarget
from dbt.exceptions import (
    CompilationError,
    DbtProfileError,
    DbtProjectError,
    DbtRuntimeError,
    ProfileConfigError,
)
from dbt.flags import get_flags
from dbt_common.clients.system import load_file_contents
from dbt_common.dataclass_schema import ValidationError
from dbt_common.events.functions import fire_event
from dbt_common.exceptions import DbtValidationError

from .renderer import ProfileRenderer

DEFAULT_THREADS: int = 1

INVALID_PROFILE_MESSAGE: str = """
dbt encountered an error while trying to read your profiles.yml file.

{error_string}
"""


def read_profile(profiles_dir: str) -> Dict[str, Any]:
    path: str = os.path.join(profiles_dir, "profiles.yml")

    contents: Optional[str] = None
    if os.path.isfile(path):
        try:
            contents = load_file_contents(path, strip=False)
            yaml_content: Dict[str, Any] = load_yaml_text(contents)
            if not yaml_content:
                msg: str = f"The profiles.yml file at {path} is empty"
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
    profile_env_vars: Dict[str, Any]
    log_cache_events: bool

    def __init__(
        self,
        profile_name: str,
        target_name: str,
        threads: int,
        credentials: Credentials,
    ) -> None:
        """Explicitly defining `__init__` to work around bug in Python 3.9.7
        https://bugs.python.org/issue45081
        """
        self.profile_name = profile_name
        self.target_name = target_name
        self.threads = threads
        self.credentials = credentials
        self.profile_env_vars = {}  # never available on init
        self.log_cache_events = get_flags().LOG_CACHE_EVENTS  # never available on init

    def to_profile_info(self, serialize_credentials: bool = False) -> Dict[str, Any]:
        """Unlike to_project_config, this dict is not a mirror of any existing
        on-disk data structure. It's used when creating a new profile from an
        existing one.
        """
        result: Dict[str, Any] = {
            "profile_name": self.profile_name,
            "target_name": self.target_name,
            "threads": self.threads,
            "credentials": self.credentials,
        }
        if serialize_credentials:
            result["credentials"] = self.credentials.to_dict(omit_none=True)
        return result

    def to_target_dict(self) -> Dict[str, Any]:
        target: Dict[str, Any] = dict(self.credentials.connection_info(with_aliases=True))
        target.update(
            {
                "type": self.credentials.type,
                "threads": self.threads,
                "name": self.target_name,
                "target_name": self.target_name,
                "profile_name": self.profile_name,
            }
        )
        return target

    def __eq__(self, other: object) -> bool:
        if not (isinstance(other, self.__class__) and isinstance(self, other.__class__)):
            return NotImplemented
        return self.to_profile_info() == other.to_profile_info()

    def validate(self) -> None:
        try:
            if self.credentials:
                dct: Dict[str, Any] = self.credentials.to_dict(omit_none=True)
                self.credentials.validate(dct)
            dct = self.to_profile_info(serialize_credentials=True)
            ProfileConfig.validate(dct)
        except ValidationError as exc:
            raise ProfileConfigError(exc) from exc

    @staticmethod
    def _credentials_from_profile(
        profile: Dict[str, Any], profile_name: str, target_name: str
    ) -> Credentials:
        from dbt.adapters.factory import load_plugin

        if "type" not in profile:
            raise DbtProfileError(
                'required field "type" not found in profile {} and target {}'.format(
                    profile_name, target_name
                )
            )

        typename: Any = profile.pop("type")
        try:
            cls_plugin = load_plugin(typename)
            data: Dict[str, Any] = cls_plugin.translate_aliases(profile)
            cls_plugin.validate(data)
            credentials: Credentials = cls_plugin.from_dict(data)
        except (DbtRuntimeError, ValidationError) as e:
            msg: str = str(e) if isinstance(e, DbtRuntimeError) else e.message
            raise DbtProfileError(
                'Credentials in profile "{}", target "{}" invalid: {}'.format(
                    profile_name, target_name, msg
                )
            ) from e

        return credentials

    @staticmethod
    def pick_profile_name(
        args_profile_name: Optional[str],
        project_profile_name: Optional[str] = None,
    ) -> str:
        def default_profiles_dir() -> Any:
            from pathlib import Path

            return Path.cwd() if (Path.cwd() / "profiles.yml").exists() else Path.home() / ".dbt"

        profile_name: Optional[str] = project_profile_name
        if args_profile_name is not None:
            profile_name = args_profile_name
        if profile_name is None:
            NO_SUPPLIED_PROFILE_ERROR: str = """\
dbt cannot run because no profile was specified for this dbt project.
To specify a profile for this project, add a line like the this to
your dbt_project.yml file:

profile: [profile name]

Here, [profile name] should be replaced with a profile name
defined in your profiles.yml file. You can find profiles.yml here:

{profiles_file}/profiles.yml
""".format(
                profiles_file=default_profiles_dir()
            )
            raise DbtProjectError(NO_SUPPLIED_PROFILE_ERROR)
        return profile_name

    @staticmethod
    def _get_profile_data(
        profile: Dict[str, Any], profile_name: str, target_name: str
    ) -> Dict[str, Any]:
        if "outputs" not in profile:
            raise DbtProfileError("outputs not specified in profile '{}'".format(profile_name))
        outputs: Any = profile["outputs"]

        if target_name not in outputs:
            outputs_list: str = "\n".join(" - {}".format(output) for output in outputs)
            msg: str = (
                "The profile '{}' does not have a target named '{}'. The "
                "valid target names for this profile are:\n{}".format(
                    profile_name, target_name, outputs_list
                )
            )
            raise DbtProfileError(msg, result_type="invalid_target")
        profile_data: Any = outputs[target_name]

        if not isinstance(profile_data, dict):
            msg: str = (
                f"output '{target_name}' of profile '{profile_name}' is "
                f"misconfigured in profiles.yml"
            )
            raise DbtProfileError(msg, result_type="invalid_target")

        return profile_data

    @classmethod
    def from_credentials(
        cls,
        credentials: Credentials,
        threads: int,
        profile_name: str,
        target_name: str,
    ) -> "Profile":
        profile: Profile = cls(
            profile_name=profile_name,
            target_name=target_name,
            threads=threads,
            credentials=credentials,
        )
        profile.validate()
        return profile

    @classmethod
    def render_profile(
        cls,
        raw_profile: Dict[str, Any],
        profile_name: str,
        target_override: Optional[str],
        renderer: ProfileRenderer,
    ) -> Tuple[str, Dict[str, Any]]:
        if target_override is not None:
            target_name: str = target_override
        elif "target" in raw_profile:
            target_name = renderer.render_value(raw_profile["target"])
        else:
            target_name = "default"
            fire_event(MissingProfileTarget(profile_name=profile_name, target_name=target_name))

        raw_profile_data: Dict[str, Any] = cls._get_profile_data(raw_profile, profile_name, target_name)

        try:
            profile_data: Dict[str, Any] = renderer.render_data(raw_profile_data)
        except CompilationError as exc:
            raise DbtProfileError(str(exc)) from exc
        return target_name, profile_data

    @classmethod
    def from_raw_profile_info(
        cls,
        raw_profile: Dict[str, Any],
        profile_name: str,
        renderer: ProfileRenderer,
        target_override: Optional[str] = None,
        threads_override: Optional[int] = None,
    ) -> "Profile":
        target_name: str
        profile_data: Dict[str, Any]
        target_name, profile_data = cls.render_profile(
            raw_profile, profile_name, target_override, renderer
        )

        threads: int = profile_data.pop("threads", DEFAULT_THREADS)
        if threads_override is not None:
            threads = threads_override

        credentials: Credentials = cls._credentials_from_profile(
            profile_data, profile_name, target_name
        )

        return cls.from_credentials(
            credentials=credentials,
            profile_name=profile_name,
            target_name=target_name,
            threads=threads,
        )

    @classmethod
    def from_raw_profiles(
        cls,
        raw_profiles: Dict[str, Any],
        profile_name: str,
        renderer: ProfileRenderer,
        target_override: Optional[str] = None,
        threads_override: Optional[int] = None,
    ) -> "Profile":
        if profile_name not in raw_profiles:
            raise DbtProjectError("Could not find profile named '{}'".format(profile_name))

        raw_profile: Any = raw_profiles[profile_name]
        if not raw_profile:
            msg: str = f"Profile {profile_name} in profiles.yml is empty"
            raise DbtProfileError(INVALID_PROFILE_MESSAGE.format(error_string=msg))

        return cls.from_raw_profile_info(
            raw_profile=raw_profile,
            profile_name=profile_name,
            renderer=renderer,
            target_override=target_override,
            threads_override=threads_override,
        )

    @classmethod
    def render(
        cls,
        renderer: ProfileRenderer,
        project_profile_name: Optional[str],
        profile_name_override: Optional[str] = None,
        target_override: Optional[str] = None,
        threads_override: Optional[int] = None,
    ) -> "Profile":
        flags = get_flags()
        raw_profiles: Dict[str, Any] = read_profile(flags.PROFILES_DIR)
        profile_name: str = cls.pick_profile_name(profile_name_override, project_profile_name)
        return cls.from_raw_profiles(
            raw_profiles=raw_profiles,
            profile_name=profile_name,
            renderer=renderer,
            target_override=target_override,
            threads_override=threads_override,
        )