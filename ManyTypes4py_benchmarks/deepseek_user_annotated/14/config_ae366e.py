"""Module to help with parsing and generating configuration files."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from functools import partial, reduce
import logging
import operator
import os
from pathlib import Path
import re
import shutil
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from awesomeversion import AwesomeVersion
import voluptuous as vol
from voluptuous.humanize import MAX_VALIDATION_ERROR_ITEM_LENGTH
from yaml.error import MarkedYAMLError

from .const import CONF_PACKAGES, CONF_PLATFORM, __version__
from .core import DOMAIN as HOMEASSISTANT_DOMAIN, HomeAssistant, callback
from .core_config import _PACKAGE_DEFINITION_SCHEMA, _PACKAGES_CONFIG_SCHEMA
from .exceptions import ConfigValidationError, HomeAssistantError
from .helpers import config_validation as cv
from .helpers.translation import async_get_exception_message
from .helpers.typing import ConfigType
from .loader import ComponentProtocol, Integration, IntegrationNotFound
from .requirements import RequirementsNotFound, async_get_integration_with_requirements
from .util.async_ import create_eager_task
from .util.package import is_docker_env
from .util.yaml import SECRET_YAML, Secrets, YamlTypeError, load_yaml_dict
from .util.yaml.objects import NodeStrClass

_LOGGER = logging.getLogger(__name__)

RE_YAML_ERROR = re.compile(r"homeassistant\.util\.yaml")
RE_ASCII = re.compile(r"\033\[[^m]*m")
YAML_CONFIG_FILE = "configuration.yaml"
VERSION_FILE = ".HA_VERSION"
CONFIG_DIR_NAME = ".homeassistant"

AUTOMATION_CONFIG_PATH = "automations.yaml"
SCRIPT_CONFIG_PATH = "scripts.yaml"
SCENE_CONFIG_PATH = "scenes.yaml"

LOAD_EXCEPTIONS = (ImportError, FileNotFoundError)
INTEGRATION_LOAD_EXCEPTIONS = (IntegrationNotFound, RequirementsNotFound)

SAFE_MODE_FILENAME = "safe-mode"

DEFAULT_CONFIG = f"""
# Loads default set of integrations. Do not remove.
default_config:

# Load frontend themes from the themes folder
frontend:
  themes: !include_dir_merge_named themes

automation: !include {AUTOMATION_CONFIG_PATH}
script: !include {SCRIPT_CONFIG_PATH}
scene: !include {SCENE_CONFIG_PATH}
"""
DEFAULT_SECRETS = """
# Use this file to store secrets like usernames and passwords.
# Learn more at https://www.home-assistant.io/docs/configuration/secrets/
some_password: welcome
"""
TTS_PRE_92 = """
tts:
  - platform: google
"""
TTS_92 = """
tts:
  - platform: google_translate
    service_name: google_say
"""


class ConfigErrorTranslationKey(StrEnum):
    """Config error translation keys for config errors."""

    CONFIG_VALIDATION_ERR = "config_validation_err"
    PLATFORM_CONFIG_VALIDATION_ERR = "platform_config_validation_err"
    COMPONENT_IMPORT_ERR = "component_import_err"
    CONFIG_PLATFORM_IMPORT_ERR = "config_platform_import_err"
    CONFIG_VALIDATOR_UNKNOWN_ERR = "config_validator_unknown_err"
    CONFIG_SCHEMA_UNKNOWN_ERR = "config_schema_unknown_err"
    PLATFORM_COMPONENT_LOAD_ERR = "platform_component_load_err"
    PLATFORM_COMPONENT_LOAD_EXC = "platform_component_load_exc"
    PLATFORM_SCHEMA_VALIDATOR_ERR = "platform_schema_validator_err"
    MULTIPLE_INTEGRATION_CONFIG_ERRORS = "multiple_integration_config_errors"


_CONFIG_LOG_SHOW_STACK_TRACE: Dict[ConfigErrorTranslationKey, bool] = {
    ConfigErrorTranslationKey.COMPONENT_IMPORT_ERR: False,
    ConfigErrorTranslationKey.CONFIG_PLATFORM_IMPORT_ERR: False,
    ConfigErrorTranslationKey.CONFIG_VALIDATOR_UNKNOWN_ERR: True,
    ConfigErrorTranslationKey.CONFIG_SCHEMA_UNKNOWN_ERR: True,
    ConfigErrorTranslationKey.PLATFORM_COMPONENT_LOAD_ERR: False,
    ConfigErrorTranslationKey.PLATFORM_COMPONENT_LOAD_EXC: True,
    ConfigErrorTranslationKey.PLATFORM_SCHEMA_VALIDATOR_ERR: True,
}


@dataclass
class ConfigExceptionInfo:
    """Configuration exception info class."""

    exception: Exception
    translation_key: ConfigErrorTranslationKey
    platform_path: str
    config: ConfigType
    integration_link: Optional[str]


@dataclass
class IntegrationConfigInfo:
    """Configuration for an integration and exception information."""

    config: Optional[ConfigType]
    exception_info_list: List[ConfigExceptionInfo]


def get_default_config_dir() -> str:
    """Put together the default configuration directory based on the OS."""
    data_dir = os.path.expanduser("~")
    return os.path.join(data_dir, CONFIG_DIR_NAME)


async def async_ensure_config_exists(hass: HomeAssistant) -> bool:
    """Ensure a configuration file exists in given configuration directory."""
    config_path = hass.config.path(YAML_CONFIG_FILE)

    if os.path.isfile(config_path):
        return True

    print(  # noqa: T201
        "Unable to find configuration. Creating default one in", hass.config.config_dir
    )
    return await async_create_default_config(hass)


async def async_create_default_config(hass: HomeAssistant) -> bool:
    """Create a default configuration file in given configuration directory."""
    return await hass.async_add_executor_job(
        _write_default_config, hass.config.config_dir
    )


def _write_default_config(config_dir: str) -> bool:
    """Write the default config."""
    config_path = os.path.join(config_dir, YAML_CONFIG_FILE)
    secret_path = os.path.join(config_dir, SECRET_YAML)
    version_path = os.path.join(config_dir, VERSION_FILE)
    automation_yaml_path = os.path.join(config_dir, AUTOMATION_CONFIG_PATH)
    script_yaml_path = os.path.join(config_dir, SCRIPT_CONFIG_PATH)
    scene_yaml_path = os.path.join(config_dir, SCENE_CONFIG_PATH)

    try:
        with open(config_path, "w", encoding="utf8") as config_file:
            config_file.write(DEFAULT_CONFIG)

        if not os.path.isfile(secret_path):
            with open(secret_path, "w", encoding="utf8") as secret_file:
                secret_file.write(DEFAULT_SECRETS)

        with open(version_path, "w", encoding="utf8") as version_file:
            version_file.write(__version__)

        if not os.path.isfile(automation_yaml_path):
            with open(automation_yaml_path, "w", encoding="utf8") as automation_file:
                automation_file.write("[]")

        if not os.path.isfile(script_yaml_path):
            with open(script_yaml_path, "w", encoding="utf8"):
                pass

        if not os.path.isfile(scene_yaml_path):
            with open(scene_yaml_path, "w", encoding="utf8"):
                pass
    except OSError:
        print(  # noqa: T201
            f"Unable to create default configuration file {config_path}"
        )
        return False
    return True


async def async_hass_config_yaml(hass: HomeAssistant) -> Dict[Any, Any]:
    """Load YAML from a Home Assistant configuration file."""
    secrets = Secrets(Path(hass.config.config_dir))

    try:
        config = await hass.loop.run_in_executor(
            None,
            load_yaml_config_file,
            hass.config.path(YAML_CONFIG_FILE),
            secrets,
        )
    except HomeAssistantError as exc:
        if not (base_exc := exc.__cause__) or not isinstance(base_exc, MarkedYAMLError):
            raise

        if base_exc.context_mark and base_exc.context_mark.name:
            base_exc.context_mark.name = _relpath(hass, base_exc.context_mark.name)
        if base_exc.problem_mark and base_exc.problem_mark.name:
            base_exc.problem_mark.name = _relpath(hass, base_exc.problem_mark.name)
        raise

    invalid_domains = []
    for key in config:
        try:
            cv.domain_key(key)
        except vol.Invalid as exc:
            suffix = ""
            if annotation := find_annotation(config, exc.path):
                suffix = f" at {_relpath(hass, annotation[0])}, line {annotation[1]}"
            _LOGGER.error("Invalid domain '%s'%s", key, suffix)
            invalid_domains.append(key)
    for invalid_domain in invalid_domains:
        config.pop(invalid_domain)

    core_config = config.get(HOMEASSISTANT_DOMAIN, {})
    try:
        await merge_packages_config(hass, config, core_config.get(CONF_PACKAGES, {}))
    except vol.Invalid as exc:
        suffix = ""
        if annotation := find_annotation(
            config, [HOMEASSISTANT_DOMAIN, CONF_PACKAGES, *exc.path]
        ):
            suffix = f" at {_relpath(hass, annotation[0])}, line {annotation[1]}"
        _LOGGER.error(
            "Invalid package configuration '%s'%s: %s", CONF_PACKAGES, suffix, exc
        )
        core_config[CONF_PACKAGES] = {}

    return config


def load_yaml_config_file(
    config_path: str, secrets: Optional[Secrets] = None
) -> Dict[Any, Any]:
    """Parse a YAML configuration file."""
    try:
        conf_dict = load_yaml_dict(config_path, secrets)
    except YamlTypeError as exc:
        msg = (
            f"The configuration file {os.path.basename(config_path)} "
            "does not contain a dictionary"
        )
        _LOGGER.error(msg)
        raise HomeAssistantError(msg) from exc

    for key, value in conf_dict.items():
        conf_dict[key] = value or {}
    return conf_dict


def process_ha_config_upgrade(hass: HomeAssistant) -> None:
    """Upgrade configuration if necessary."""
    version_path = hass.config.path(VERSION_FILE)

    try:
        with open(version_path, encoding="utf8") as inp:
            conf_version = inp.readline().strip()
    except FileNotFoundError:
        conf_version = "0.7.7"

    if conf_version == __version__:
        return

    _LOGGER.info(
        "Upgrading configuration directory from %s to %s", conf_version, __version__
    )

    version_obj = AwesomeVersion(conf_version)

    if version_obj < AwesomeVersion("0.50"):
        lib_path = hass.config.path("deps")
        if os.path.isdir(lib_path):
            shutil.rmtree(lib_path)

    if version_obj < AwesomeVersion("0.92"):
        config_path = hass.config.path(YAML_CONFIG_FILE)

        with open(config_path, encoding="utf-8") as config_file:
            config_raw = config_file.read()

        if TTS_PRE_92 in config_raw:
            _LOGGER.info("Migrating google tts to google_translate tts")
            config_raw = config_raw.replace(TTS_PRE_92, TTS_92)
            try:
                with open(config_path, "w", encoding="utf-8") as config_file:
                    config_file.write(config_raw)
            except OSError:
                _LOGGER.exception("Migrating to google_translate tts failed")

    if version_obj < AwesomeVersion("0.94") and is_docker_env():
        lib_path = hass.config.path("deps")
        if os.path.isdir(lib_path):
            shutil.rmtree(lib_path)

    with open(version_path, "w", encoding="utf8") as outp:
        outp.write(__version__)


@callback
def async_log_schema_error(
    exc: vol.Invalid,
    domain: str,
    config: Dict[str, Any],
    hass: HomeAssistant,
    link: Optional[str] = None,
) -> None:
    """Log a schema validation error."""
    message = format_schema_error(hass, exc, domain, config, link)
    _LOGGER.error(message)


@callback
def async_log_config_validator_error(
    exc: Union[vol.Invalid, HomeAssistantError],
    domain: str,
    config: Dict[str, Any],
    hass: HomeAssistant,
    link: Optional[str] = None,
) -> None:
    """Log an error from a custom config validator."""
    if isinstance(exc, vol.Invalid):
        async_log_schema_error(exc, domain, config, hass, link)
        return

    message = format_homeassistant_error(hass, exc, domain, config, link)
    _LOGGER.error(message, exc_info=exc)


def _get_annotation(item: Any) -> Optional[Tuple[str, Union[int, str]]]:
    if not hasattr(item, "__config_file__"):
        return None

    return (getattr(item, "__config_file__"), getattr(item, "__line__", "?"))


def _get_by_path(data: Union[Dict[Any, Any], List[Any]], items: List[Hashable]) -> Any:
    """Access a nested object in root by item sequence."""
    try:
        return reduce(operator.getitem, items, data)  # type: ignore[arg-type]
    except (KeyError, IndexError, TypeError):
        return None


def find_annotation(
    config: Union[Dict[Any, Any], List[Any]], path: List[Hashable]
) -> Optional[Tuple[str, Union[int, str]]]:
    """Find file/line annotation for a node in config pointed to by path."""

    def find_annotation_for_key(
        item: Dict[Any, Any], path: List[Hashable], tail: Hashable
    ) -> Optional[Tuple[str, Union[int, str]]]:
        for key in item:
            if key == tail:
                if annotation := _get_annotation(key):
                    return annotation
                break
        return None

    def find_annotation_rec(
        config: Union[Dict[Any, Any], List[Any]], path: List[Hashable], tail: Optional[Hashable]
    ) -> Optional[Tuple[str, Union[int, str]]]:
        item = _get_by_path(config, path)
        if isinstance(item, Dict) and tail is not None:
            if tail_annotation := find_annotation_for_key(item, path, tail):
                return tail_annotation

        if (
            isinstance(item, (Dict, List))
            and path
            and (
                key_annotation := find_annotation_for_key(
                    _get_by_path(config, path[:-1]), path[:-1], path[-1]
                )
            )
        ):
            return key_annotation

        if annotation := _get_annotation(item):
            return annotation

        if not path:
            return None

        tail = path.pop()
        if annotation := find_annotation_rec(config, path, tail):
            return annotation
        return _get_annotation(item)

    return find_annotation_rec(config, list(path), None)


def _relpath(hass: HomeAssistant, path: str) -> str:
    """Return path relative to the Home Assistant config dir."""
    return os.path.relpath(path, hass.config.config_dir)


def stringify_invalid(
    hass: HomeAssistant,
    exc: vol.Invalid,
    domain: str,
    config: Dict[str, Any],
    link: Optional[str],
    max_sub_error_length: int,
) -> str:
    """Stringify voluptuous.Invalid."""
    if "." in domain:
        integration_domain, _, platform_domain = domain.partition(".")
        message_prefix = (
            f"Invalid config for '{platform_domain}' from integration "
            f"'{integration_domain}'"
        )
    else:
        message_prefix = f"Invalid config for '{domain}'"
    if domain != HOMEASSISTANT_DOMAIN and link:
        message_suffix = f", please check the docs at {link}"
    else:
        message_suffix = ""
    if annotation := find_annotation(config, exc.path):
        message_prefix += f" at {_relpath(hass, annotation[0])}, line {annotation[1]}"
    path = "->".join(str(m) for m in exc.path)
    if exc.error_message == "extra keys not allowed":
        return (
            f"{message_prefix}: '{exc.path[-1]}' is an invalid option for '{domain}', "
            f"check: {path}{message_suffix}"
        )
    if exc.error_message == "required key not provided":
        return (
            f"{message_prefix}: required key '{exc.path[-1]}' not provided"
            f"{message_suffix}"
        )
    output = Exception.__str__(exc)
    if error_type := exc.error_type:
        output += " for " + error_type
    offending_item_summary = repr(_get_by_path(config, exc.path))
    if len(offending_item_summary) > max_sub_error_length:
        offending_item_summary = (
            f"{offending_item_summary[: max_sub_error_length - 3]}..."
        )
    return (
        f"{message_prefix}: {output} '{path}', got {offending_item_summary}"
        f"{message_suffix}"
    )


def humanize_error(
    hass: HomeAssistant,
    validation_error: vol.Invalid,
    domain: str,
    config: Dict[str, Any],
    link: Optional[str],
    max_sub_error_length: int = MAX_VALIDATION_ERROR_ITEM_LENGTH,
) -> str:
    """Provide a more helpful + complete validation error message."""
    if isinstance(validation_error, vol.MultipleInvalid):
        return "\n".join(
            sorted(
                humanize_error(
                    hass, sub_error, domain, config, link, max_sub_error_length
                )
                for sub_error in validation_error.errors
            )
        )
    return stringify_invalid(
        hass, validation_error, domain, config, link, max_sub_error_length
    )


@callback
def format_homeassistant_error(
    hass: HomeAssistant,
    exc: HomeAssistantError,
    domain: str,
    config: