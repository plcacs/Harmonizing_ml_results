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
from typing import TYPE_CHECKING, Any
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

_LOGGER: logging.Logger = logging.getLogger(__name__)
RE_YAML_ERROR: re.Pattern = re.compile('homeassistant\\.util\\.yaml')
RE_ASCII: re.Pattern = re.compile('\\033\\[[^m]*m')
YAML_CONFIG_FILE: str = 'configuration.yaml'
VERSION_FILE: str = '.HA_VERSION'
CONFIG_DIR_NAME: str = '.homeassistant'
AUTOMATION_CONFIG_PATH: str = 'automations.yaml'
SCRIPT_CONFIG_PATH: str = 'scripts.yaml'
SCENE_CONFIG_PATH: str = 'scenes.yaml'
LOAD_EXCEPTIONS: tuple = (ImportError, FileNotFoundError)
INTEGRATION_LOAD_EXCEPTIONS: tuple = (IntegrationNotFound, RequirementsNotFound)
SAFE_MODE_FILENAME: str = 'safe-mode'
DEFAULT_CONFIG: str = f'\n# Loads default set of integrations. Do not remove.\ndefault_config:\n\n# Load frontend themes from the themes folder\nfrontend:\n  themes: !include_dir_merge_named themes\n\nautomation: !include {AUTOMATION_CONFIG_PATH}\nscript: !include {SCRIPT_CONFIG_PATH}\nscene: !include {SCENE_CONFIG_PATH}\n'
DEFAULT_SECRETS: str = '\n# Use this file to store secrets like usernames and passwords.\n# Learn more at https://www.home-assistant.io/docs/configuration/secrets/\nsome_password: welcome\n'
TTS_PRE_92: str = '\ntts:\n  - platform: google\n'
TTS_92: str = '\ntts:\n  - platform: google_translate\n    service_name: google_say\n'

@dataclass
class ConfigExceptionInfo:
    """Configuration exception info class."""

@dataclass
class IntegrationConfigInfo:
    """Configuration for an integration and exception information."""

def get_default_config_dir() -> str:
    """Put together the default configuration directory based on the OS."""
    data_dir: str = os.path.expanduser('~')
    return os.path.join(data_dir, CONFIG_DIR_NAME)

async def async_ensure_config_exists(hass: HomeAssistant) -> bool:
    """Ensure a configuration file exists in given configuration directory.

    Creating a default one if needed.
    Return boolean if configuration dir is ready to go.
    """
    config_path: str = hass.config.path(YAML_CONFIG_FILE)
    if os.path.isfile(config_path):
        return True
    print('Unable to find configuration. Creating default one in', hass.config.config_dir)
    return await async_create_default_config(hass)

async def async_create_default_config(hass: HomeAssistant) -> bool:
    """Create a default configuration file in given configuration directory.

    Return if creation was successful.
    """
    return await hass.async_add_executor_job(_write_default_config, hass.config.config_dir)

def _write_default_config(config_dir: str) -> bool:
    """Write the default config."""
    config_path: str = os.path.join(config_dir, YAML_CONFIG_FILE)
    secret_path: str = os.path.join(config_dir, SECRET_YAML)
    version_path: str = os.path.join(config_dir, VERSION_FILE)
    automation_yaml_path: str = os.path.join(config_dir, AUTOMATION_CONFIG_PATH)
    script_yaml_path: str = os.path.join(config_dir, SCRIPT_CONFIG_PATH)
    scene_yaml_path: str = os.path.join(config_dir, SCENE_CONFIG_PATH)
    try:
        with open(config_path, 'w', encoding='utf8') as config_file:
            config_file.write(DEFAULT_CONFIG)
        if not os.path.isfile(secret_path):
            with open(secret_path, 'w', encoding='utf8') as secret_file:
                secret_file.write(DEFAULT_SECRETS)
        with open(version_path, 'w', encoding='utf8') as version_file:
            version_file.write(__version__)
        if not os.path.isfile(automation_yaml_path):
            with open(automation_yaml_path, 'w', encoding='utf8') as automation_file:
                automation_file.write('[]')
        if not os.path.isfile(script_yaml_path):
            with open(script_yaml_path, 'w', encoding='utf8'):
                pass
        if not os.path.isfile(scene_yaml_path):
            with open(scene_yaml_path, 'w', encoding='utf8'):
                pass
    except OSError:
        print(f'Unable to create default configuration file {config_path}')
        return False
    return True

async def async_hass_config_yaml(hass: HomeAssistant) -> ConfigType:
    """Load YAML from a Home Assistant configuration file.

    This function allows a component inside the asyncio loop to reload its
    configuration by itself. Include package merge.
    """
    secrets: Secrets = Secrets(Path(hass.config.config_dir))
    try:
        config: ConfigType = await hass.loop.run_in_executor(None, load_yaml_config_file, hass.config.path(YAML_CONFIG_FILE), secrets)
    except HomeAssistantError as exc:
        if not (base_exc := exc.__cause__) or not isinstance(base_exc, MarkedYAMLError):
            raise
        if base_exc.context_mark and base_exc.context_mark.name:
            base_exc.context_mark.name = _relpath(hass, base_exc.context_mark.name)
        if base_exc.problem_mark and base_exc.problem_mark.name:
            base_exc.problem_mark.name = _relpath(hass, base_exc.problem_mark.name)
        raise
    invalid_domains: list = []
    for key in config:
        try:
            cv.domain_key(key)
        except vol.Invalid as exc:
            suffix: str = ''
            if (annotation := find_annotation(config, exc.path)):
                suffix = f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
            _LOGGER.error("Invalid domain '%s'%s", key, suffix)
            invalid_domains.append(key)
    for invalid_domain in invalid_domains:
        config.pop(invalid_domain)
    core_config: dict = config.get(HOMEASSISTANT_DOMAIN, {})
    try:
        await merge_packages_config(hass, config, core_config.get(CONF_PACKAGES, {}))
    except vol.Invalid as exc:
        suffix: str = ''
        if (annotation := find_annotation(config, [HOMEASSISTANT_DOMAIN, CONF_PACKAGES, *exc.path])):
            suffix = f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
        _LOGGER.error("Invalid package configuration '%s'%s: %s", CONF_PACKAGES, suffix, exc)
        core_config[CONF_PACKAGES] = {}
    return config

def load_yaml_config_file(config_path: str, secrets: Secrets = None) -> ConfigType:
    """Parse a YAML configuration file.

    Raises FileNotFoundError or HomeAssistantError.

    This method needs to run in an executor.
    """
    try:
        conf_dict: ConfigType = load_yaml_dict(config_path, secrets)
    except YamlTypeError as exc:
        msg: str = f'The configuration file {os.path.basename(config_path)} does not contain a dictionary'
        _LOGGER.error(msg)
        raise HomeAssistantError(msg) from exc
    for key, value in conf_dict.items():
        conf_dict[key] = value or {}
    return conf_dict

def process_ha_config_upgrade(hass: HomeAssistant):
    """Upgrade configuration if necessary.

    This method needs to run in an executor.
    """
    version_path: str = hass.config.path(VERSION_FILE)
    try:
        with open(version_path, encoding='utf8') as inp:
            conf_version: str = inp.readline().strip()
    except FileNotFoundError:
        conf_version: str = '0.7.7'
    if conf_version == __version__:
        return
    _LOGGER.info('Upgrading configuration directory from %s to %s', conf_version, __version__)
    version_obj: AwesomeVersion = AwesomeVersion(conf_version)
    if version_obj < AwesomeVersion('0.50'):
        lib_path: str = hass.config.path('deps')
        if os.path.isdir(lib_path):
            shutil.rmtree(lib_path)
    if version_obj < AwesomeVersion('0.92'):
        config_path: str = hass.config.path(YAML_CONFIG_FILE)
        with open(config_path, encoding='utf-8') as config_file:
            config_raw: str = config_file.read()
        if TTS_PRE_92 in config_raw:
            _LOGGER.info('Migrating google tts to google_translate tts')
            config_raw: str = config_raw.replace(TTS_PRE_92, TTS_92)
            try:
                with open(config_path, 'w', encoding='utf-8') as config_file:
                    config_file.write(config_raw)
            except OSError:
                _LOGGER.exception('Migrating to google_translate tts failed')
    if version_obj < AwesomeVersion('0.94') and is_docker_env():
        lib_path: str = hass.config.path('deps')
        if os.path.isdir(lib_path):
            shutil.rmtree(lib_path)
    with open(version_path, 'w', encoding='utf8') as outp:
        outp.write(__version__)

@callback
def async_log_schema_error(exc: vol.Invalid, domain: str, config: ConfigType, hass: HomeAssistant, link: str = None) -> None:
    """Log a schema validation error."""
    message: str = format_schema_error(hass, exc, domain, config, link)
    _LOGGER.error(message)

@callback
def async_log_config_validator_error(exc: Exception, domain: str, config: ConfigType, hass: HomeAssistant, link: str = None) -> None:
    """Log an error from a custom config validator."""
    if isinstance(exc, vol.Invalid):
        async_log_schema_error(exc, domain, config, hass, link)
        return
    message: str = format_homeassistant_error(hass, exc, domain, config, link)
    _LOGGER.error(message, exc_info=exc)

def _get_annotation(item: Any) -> tuple:
    if not hasattr(item, '__config_file__'):
        return None
    return (getattr(item, '__config_file__'), getattr(item, '__line__', '?'))

def _get_by_path(data: Any, items: list) -> Any:
    """Access a nested object in root by item sequence.

    Returns None in case of error.
    """
    try:
        return reduce(operator.getitem, items, data)
    except (KeyError, IndexError, TypeError):
        return None

def find_annotation(config: ConfigType, path: list) -> tuple:
    """Find file/line annotation for a node in config pointed to by path.

    If the node pointed to is a dict or list, prefer the annotation for the key in
    the key/value pair defining the dict or list.
    If the node is not annotated, try the parent node.
    """

    def find_annotation_for_key(item: Any, path: list, tail: Any) -> tuple:
        for key in item:
            if key == tail:
                if (annotation := _get_annotation(key)):
                    return annotation
                break
        return None

    def find_annotation_rec(config: ConfigType, path: list, tail: Any) -> tuple:
        item: Any = _get_by_path(config, path)
        if isinstance(item, dict) and tail is not None:
            if (tail_annotation := find_annotation_for_key(item, path, tail)):
                return tail_annotation
        if isinstance(item, (dict, list)) and path and (key_annotation := find_annotation_for_key(_get_by_path(config, path[:-1]), path[:-1], path[-1])):
            return key_annotation
        if (annotation := _get_annotation(item)):
            return annotation
        if not path:
            return None
        tail = path.pop()
        if (annotation := find_annotation_rec(config, path, tail)):
            return annotation
        return _get_annotation(item)
    return find_annotation_rec(config, list(path), None)

def _relpath(hass: HomeAssistant, path: str) -> str:
    """Return path relative to the Home Assistant config dir."""
    return os.path.relpath(path, hass.config.config_dir)

def stringify_invalid(hass: HomeAssistant, exc: vol.Invalid, domain: str, config: ConfigType, link: str, max_sub_error_length: int) -> str:
    """Stringify voluptuous.Invalid.

    This is an alternative to the custom __str__ implemented in
    voluptuous.error.Invalid. The modifications are:
    - Format the path delimited by -> instead of @data[]
    - Prefix with domain, file and line of the error
    - Suffix with a link to the documentation
    - Give a more user friendly output for unknown options
    - Give a more user friendly output for missing options
    """
    if '.' in domain:
        integration_domain, _, platform_domain = domain.partition('.')
        message_prefix: str = f"Invalid config for '{platform_domain}' from integration '{integration_domain}'"
    else:
        message_prefix: str = f"Invalid config for '{domain}'"
    if domain != HOMEASSISTANT_DOMAIN and link:
        message_suffix: str = f', please check the docs at {link}'
    else:
        message_suffix: str = ''
    if (annotation := find_annotation(config, exc.path)):
        message_prefix += f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
    path: str = '->'.join((str(m) for m in exc.path))
    if exc.error_message == 'extra keys not allowed':
        return f"{message_prefix}: '{exc.path[-1]}' is an invalid option for '{domain}', check: {path}{message_suffix}"
    if exc.error_message == 'required key not provided':
        return f"{message_prefix}: required key '{exc.path[-1]}' not provided{message_suffix}"
    output: str = Exception.__str__(exc)
    if (error_type := exc.error_type):
        output += ' for ' + error_type
    offending_item_summary: str = repr(_get_by_path(config, exc.path))
    if len(offending_item_summary) > max_sub_error_length:
        offending_item_summary = f'{offending_item_summary[:max_sub_error_length - 3]}...'
    return f"{message_prefix}: {output} '{path}', got {offending_item_summary}{message_suffix}"

def humanize_error(hass: HomeAssistant, validation_error: vol.MultipleInvalid, domain: str, config: ConfigType, link: str, max_sub_error_length: int = MAX_VALIDATION_ERROR_ITEM_LENGTH) -> str:
    """Provide a more helpful + complete validation error message.

    This is a modified version of voluptuous.error.Invalid.__str__,
    the modifications make some minor changes to the formatting.
    """
    if isinstance(validation_error, vol.MultipleInvalid):
        return '\n'.join(sorted((humanize_error(hass, sub_error, domain, config, link, max_sub_error_length) for sub_error in validation_error.errors)))
    return stringify_invalid(hass, validation_error, domain, config, link, max_sub_error_length)

@callback
def format_homeassistant_error(hass: HomeAssistant, exc: Exception, domain: str, config: ConfigType, link: str = None) -> str:
    """Format HomeAssistantError thrown by a custom config validator."""
    if '.' in domain:
        integration_domain, _, platform_domain = domain.partition('.')
        message_prefix: str = f"Invalid config for '{platform_domain}' from integration '{integration_domain}'"
    else:
        message_prefix: str = f"Invalid config for '{domain}'"
    if (annotation := find_annotation(config, [domain])):
        message_prefix += f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
    message: str = f'{message_prefix}: {str(exc) or repr(exc)}'
    if domain != HOMEASSISTANT_DOMAIN and link:
        message += f', please check the docs at {link}'
    return message

@callback
def format_schema_error(hass: HomeAssistant, exc: vol.Invalid, domain: str, config: ConfigType, link: str = None) -> str:
    """Format configuration validation error."""
    return humanize_error(hass, exc, domain, config, link)

def _log_pkg_error(hass: HomeAssistant, package: str, component: str, config: ConfigType, message: str) -> None:
    """Log an error while merging packages."""
    message_prefix: str = f"Setup of package '{package}'"
    if (annotation := find_annotation(config, [HOMEASSISTANT_DOMAIN, CONF_PACKAGES, package])):
        message_prefix += f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
    _LOGGER.error('%s failed: %s', message_prefix, message)

def _identify_config_schema(module: ModuleType) -> str:
    """Extract the schema and identify list or dict based."""
    if not isinstance(module.CONFIG_SCHEMA, vol.Schema):
        return None
    schema: dict = module.CONFIG_SCHEMA.schema
    if isinstance(schema, vol.All):
        for subschema in schema.validators:
            if isinstance(subschema, dict):
                schema = subschema
                break
        else:
            return None
    try:
        key: Any = next((k for k in schema if k == module.DOMAIN))
    except (TypeError, AttributeError, StopIteration):
        return None
    except Exception:
        _LOGGER.exception('Unexpected error identifying config schema')
        return None
    if hasattr(key, 'default') and (not isinstance(key.default, vol.schema_builder.Undefined)):
        default_value: Any = module.CONFIG_SCHEMA({module.DOMAIN: key.default()})[module.DOMAIN]
        if isinstance(default_value, dict):
            return 'dict'
        if isinstance(default_value, list):
            return 'list'
        return None
    domain_schema: Any = schema[key]
    t_schema: str = str(domain_schema)
    if t_schema.startswith('{') or 'schema_with_slug_keys' in t_schema:
        return 'dict'
    if t_schema.startswith(('[', 'All(<function ensure_list')):
        return 'list'
    return None

def _validate_package_definition(name: str, conf: dict) -> None:
    """Validate basic package definition properties."""
    cv.slug(name)
    _PACKAGE_DEFINITION_SCHEMA(conf