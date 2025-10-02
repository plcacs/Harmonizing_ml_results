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
from dbt_common.exceptions import DbtValidationError
from .renderer import ProfileRenderer

DEFAULT_THREADS: int = 1
INVALID_PROFILE_MESSAGE: str = (
    "\ndbt encountered an error while trying to read your profiles.yml file.\n\n{error_string}\n"
)

def read_profile(profiles_dir: str) -> Dict[str, Any]:
    path: str = os.path.join(profiles_dir, "profiles.yml")
    contents: Optional[str] = None
    if os.path.isfile(path):
        try:
            contents = load_file_contents(path, strip=False)
            yaml_content: Any = load_yaml_text(contents)
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
    # Instance attributes
    profile_name: str
    target_name: str
    threads: int
    credentials: Credentials
    profile_env_vars: Dict[str, Any]
    log_cache_events: bool

    def __init__(self, profile_name: str, target_name: str, threads: int, credentials: Credentials) -> None:
        """Explicitly defining `__init__` to work around bug in Python 3.9.7
        https://bugs.python.org/issue45081
        """
        self.profile_name = profile_name
        self.target_name = target_name
        self.threads = threads
        self.credentials = credentials
        self.profile_env_vars = {}
        self.log_cache_events = get_flags().LOG_CACHE_EVENTS

    def to_profile_info(self, serialize_credentials: bool = False) -> Dict[str, Any]:
        """Unlike to_project_config, this dict is not a mirror of any existing
        on-disk data structure. It's used when creating a new profile from an
        existing one.

        :param serialize_credentials bool: If True, serialize the credentials.
            Otherwise, the Credentials object will be copied.
        :returns dict: The serialized profile.
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
        target.update({
            "type": self.credentials.type,
            "threads": self.threads,
            "name": self.target_name,
            "target_name": self.target_name,
            "profile_name": self.profile_name,
        })
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
    def _credentials_from_profile(profile: Dict[str, Any], profile_name: str, target_name: str) -> Credentials:
        from dbt.adapters.factory import load_plugin
        if "type" not in profile:
            raise DbtProfileError(
                'required field "type" not found in profile {} and target {}'.format(profile_name, target_name)
            )
        typename: str = profile.pop("type")
        try:
            cls_plugin = load_plugin(typename)
            data: Dict[str, Any] = cls_plugin.translate_aliases(profile)
            cls_plugin.validate(data)
            credentials: Credentials = cls_plugin.from_dict(data)
        except (DbtRuntimeError, ValidationError) as e:
            msg: str = str(e) if isinstance(e, DbtRuntimeError) else e.message
            raise DbtProfileError('Credentials in profile "{}", target "{}" invalid: {}'.format(profile_name, target_name, msg)) from e
        return credentials

    @staticmethod
    def pick_profile_name(args_profile_name: Optional[str], project_profile_name: Optional[str] = None) -> str:
        from pathlib import Path

        def default_profiles_dir() -> Path:
            return Path.cwd() if (Path.cwd() / "profiles.yml").exists() else Path.home() / ".dbt"

        profile_name: Optional[str] = project_profile_name
        if args_profile_name is not None:
            profile_name = args_profile_name
        if profile_name is None:
            NO_SUPPLIED_PROFILE_ERROR: str = (
                "dbt cannot run because no profile was specified for this dbt project.\n"
                "To specify a profile for this project, add a line like the this to\n"
                "your dbt_project.yml file:\n\nprofile: [profile name]\n\n"
                "Here, [profile name] should be replaced with a profile name\n"
                "defined in your profiles.yml file. You can find profiles.yml here:\n\n{profiles_file}/profiles.yml\n".format(
                    profiles_file=default_profiles_dir()
                )
            )
            raise DbtProjectError(NO_SUPPLIED_PROFILE_ERROR)
        return profile_name

    @staticmethod
    def _get_profile_data(profile: Dict[str, Any], profile_name: str, target_name: str) -> Dict[str, Any]:
        if "outputs" not in profile:
            raise DbtProfileError("outputs not specified in profile '{}'".format(profile_name))
        outputs: Any = profile["outputs"]
        if target_name not in outputs:
            outputs_str: str = "\n".join((" - {}".format(output) for output in outputs))
            msg: str = "The profile '{}' does not have a target named '{}'. The valid target names for this profile are:\n{}".format(
                profile_name, target_name, outputs_str
            )
            raise DbtProfileError(msg, result_type="invalid_target")
        profile_data: Any = outputs[target_name]
        if not isinstance(profile_data, dict):
            msg: str = f"output '{target_name}' of profile '{profile_name}' is misconfigured in profiles.yml"
            raise DbtProfileError(msg, result_type="invalid_target")
        return profile_data

    @classmethod
    def from_credentials(cls, credentials: Credentials, threads: int, profile_name: str, target_name: str) -> "Profile":
        """Create a profile from an existing set of Credentials and the
        remaining information.

        :param credentials: The credentials dict for this profile.
        :param threads: The number of threads to use for connections.
        :param profile_name: The profile name used for this profile.
        :param target_name: The target name used for this profile.
        :raises DbtProfileError: If the profile is invalid.
        :returns: The new Profile object.
        """
        profile: Profile = cls(profile_name=profile_name, target_name=target_name, threads=threads, credentials=credentials)
        profile.validate()
        return profile

    @classmethod
    def render_profile(cls, raw_profile: Dict[str, Any], profile_name: str, target_override: Optional[str], renderer: ProfileRenderer) -> Tuple[str, Dict[str, Any]]:
        """This is a containment zone for the hateful way we're rendering
        profiles.
        """
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
        return (target_name, profile_data)

    @classmethod
    def from_raw_profile_info(
        cls,
        raw_profile: Dict[str, Any],
        profile_name: str,
        renderer: ProfileRenderer,
        target_override: Optional[str] = None,
        threads_override: Optional[int] = None,
    ) -> "Profile":
        """Create a profile from its raw profile information.

         (this is an intermediate step, mostly useful for unit testing)

        :param raw_profile: The profile data for a single profile, from
            disk as yaml and its values rendered with jinja.
        :param profile_name: The profile name used.
        :param renderer: The config renderer.
        :param target_override: The target to use, if provided on
            the command line.
        :param threads_override: The thread count to use, if
            provided on the command line.
        :raises DbtProfileError: If the profile is invalid or missing, or the
            target could not be found
        :returns: The new Profile object.
        """
        target_name, profile_data = cls.render_profile(raw_profile, profile_name, target_override, renderer)
        threads: int = profile_data.pop("threads", DEFAULT_THREADS)
        if threads_override is not None:
            threads = threads_override
        credentials: Credentials = cls._credentials_from_profile(profile_data, profile_name, target_name)
        return cls.from_credentials(credentials=credentials, profile_name=profile_name, target_name=target_name, threads=threads)

    @classmethod
    def from_raw_profiles(
        cls,
        raw_profiles: Dict[str, Any],
        profile_name: str,
        renderer: ProfileRenderer,
        target_override: Optional[str] = None,
        threads_override: Optional[int] = None,
    ) -> "Profile":
        """
        :param raw_profiles: The profile data, from disk as yaml.
        :param profile_name: The profile name to use.
        :param renderer: The config renderer.
        :param target_override: The target to use, if provided on the
            command line.
        :param threads_override: The thread count to use, if provided on the
            command line.
        :raises DbtProjectError: If there is no profile name specified in the
            project or the command line arguments
        :raises DbtProfileError: If the profile is invalid or missing, or the
            target could not be found
        :returns: The new Profile object.
        """
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
        """Given the raw profiles as read from disk and the name of the desired
        profile if specified, return the profile component of the runtime
        config.

        :param renderer: The profile renderer.
        :param project_profile_name: The profile name, if specified in a project.
        :param profile_name_override: The profile name to override, if provided.
        :param target_override: The target to override, if provided.
        :param threads_override: The thread count to override, if provided.
        :raises DbtProjectError: If there is no profile name specified in the
            project or the command line arguments, or if the specified profile
            is not found
        :raises DbtProfileError: If the profile is invalid or missing, or the
            target could not be found.
        :returns Profile: The new Profile object.
        """
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