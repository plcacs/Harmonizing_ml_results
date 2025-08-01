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
from typing import Any, Dict, List, Optional, Tuple

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
RE_YAML_ERROR = re.compile('homeassistant\\.util\\.yaml')
RE_ASCII = re.compile('\\033\\[[^m]*m')
YAML_CONFIG_FILE = 'configuration.yaml'
VERSION_FILE = '.HA_VERSION'
CONFIG_DIR_NAME = '.homeassistant'
AUTOMATION_CONFIG_PATH = 'automations.yaml'
SCRIPT_CONFIG_PATH = 'scripts.yaml'
SCENE_CONFIG_PATH = 'scenes.yaml'
LOAD_EXCEPTIONS = (ImportError, FileNotFoundError)
INTEGRATION_LOAD_EXCEPTIONS = (IntegrationNotFound, RequirementsNotFound)
SAFE_MODE_FILENAME = 'safe-mode'
DEFAULT_CONFIG = (
    f'\n# Loads default set of integrations. Do not remove.\n'
    f'default_config:\n\n'
    f'# Load frontend themes from the themes folder\n'
    f'frontend:\n  themes: !include_dir_merge_named themes\n\n'
    f'automation: !include {AUTOMATION_CONFIG_PATH}\n'
    f'script: !include {SCRIPT_CONFIG_PATH}\n'
    f'scene: !include {SCENE_CONFIG_PATH}\n'
)
DEFAULT_SECRETS = (
    '\n# Use this file to store secrets like usernames and passwords.\n'
    '# Learn more at https://www.home-assistant.io/docs/configuration/secrets/\n'
    'some_password: welcome\n'
)
TTS_PRE_92 = '\ntts:\n  - platform: google\n'
TTS_92 = (
    '\ntts:\n  - platform: google_translate\n'
    '    service_name: google_say\n'
)

class ConfigErrorTranslationKey(StrEnum):
    CONFIG_VALIDATION_ERR = 'config_validation_err'
    PLATFORM_CONFIG_VALIDATION_ERR = 'platform_config_validation_err'
    COMPONENT_IMPORT_ERR = 'component_import_err'
    CONFIG_PLATFORM_IMPORT_ERR = 'config_platform_import_err'
    CONFIG_VALIDATOR_UNKNOWN_ERR = 'config_validator_unknown_err'
    CONFIG_SCHEMA_UNKNOWN_ERR = 'config_schema_unknown_err'
    PLATFORM_COMPONENT_LOAD_ERR = 'platform_component_load_err'
    PLATFORM_COMPONENT_LOAD_EXC = 'platform_component_load_exc'
    PLATFORM_SCHEMA_VALIDATOR_ERR = 'platform_schema_validator_err'
    MULTIPLE_INTEGRATION_CONFIG_ERRORS = 'multiple_integration_config_errors'

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
    exception: Exception
    translation_key: ConfigErrorTranslationKey
    path: Any
    config: Any
    integration_docs: Optional[str] = None


@dataclass
class IntegrationConfigInfo:
    config: Optional[Dict[str, Any]]
    exception_info_list: List[ConfigExceptionInfo]


def get_default_config_dir() -> str:
    data_dir: str = os.path.expanduser('~')
    return os.path.join(data_dir, CONFIG_DIR_NAME)


async def async_ensure_config_exists(hass: HomeAssistant) -> bool:
    config_path: str = hass.config.path(YAML_CONFIG_FILE)
    if os.path.isfile(config_path):
        return True
    print('Unable to find configuration. Creating default one in', hass.config.config_dir)
    return await async_create_default_config(hass)


async def async_create_default_config(hass: HomeAssistant) -> bool:
    return await hass.async_add_executor_job(_write_default_config, hass.config.config_dir)


def _write_default_config(config_dir: str) -> bool:
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
    secrets: Secrets = Secrets(Path(hass.config.config_dir))
    try:
        config: ConfigType = await hass.loop.run_in_executor(
            None,
            load_yaml_config_file,
            hass.config.path(YAML_CONFIG_FILE),
            secrets
        )
    except HomeAssistantError as exc:
        base_exc: Optional[Exception] = exc.__cause__
        if not base_exc or not isinstance(base_exc, MarkedYAMLError):
            raise
        if base_exc.context_mark and base_exc.context_mark.name:
            base_exc.context_mark.name = _relpath(hass, base_exc.context_mark.name)
        if base_exc.problem_mark and base_exc.problem_mark.name:
            base_exc.problem_mark.name = _relpath(hass, base_exc.problem_mark.name)
        raise
    invalid_domains: List[str] = []
    for key in config:
        try:
            cv.domain_key(key)
        except vol.Invalid as exc:
            suffix: str = ''
            annotation = find_annotation(config, exc.path)
            if annotation:
                suffix = f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
            _LOGGER.error("Invalid domain '%s'%s", key, suffix)
            invalid_domains.append(key)
    for invalid_domain in invalid_domains:
        config.pop(invalid_domain)
    core_config: Dict[str, Any] = config.get(HOMEASSISTANT_DOMAIN, {})
    try:
        await merge_packages_config(hass, config, core_config.get(CONF_PACKAGES, {}))
    except vol.Invalid as exc:
        suffix = ''
        annotation = find_annotation(config, [HOMEASSISTANT_DOMAIN, CONF_PACKAGES, *exc.path])
        if annotation:
            suffix = f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
        _LOGGER.error("Invalid package configuration '%s'%s: %s", CONF_PACKAGES, suffix, exc)
        core_config[CONF_PACKAGES] = {}
    return config


def load_yaml_config_file(config_path: str, secrets: Optional[Secrets] = None) -> Dict[str, Any]:
    try:
        conf_dict: Dict[str, Any] = load_yaml_dict(config_path, secrets)
    except YamlTypeError as exc:
        msg: str = f'The configuration file {os.path.basename(config_path)} does not contain a dictionary'
        _LOGGER.error(msg)
        raise HomeAssistantError(msg) from exc
    for key, value in conf_dict.items():
        conf_dict[key] = value or {}
    return conf_dict


def process_ha_config_upgrade(hass: HomeAssistant) -> None:
    version_path: str = hass.config.path(VERSION_FILE)
    try:
        with open(version_path, encoding='utf8') as inp:
            conf_version: str = inp.readline().strip()
    except FileNotFoundError:
        conf_version = '0.7.7'
    if conf_version == __version__:
        return
    _LOGGER.info('Upgrading configuration directory from %s to %s', conf_version, __version__)
    version_obj = AwesomeVersion(conf_version)
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
            config_raw = config_raw.replace(TTS_PRE_92, TTS_92)
            try:
                with open(config_path, 'w', encoding='utf-8') as config_file:
                    config_file.write(config_raw)
            except OSError:
                _LOGGER.exception('Migrating to google_translate tts failed')
    if version_obj < AwesomeVersion('0.94') and is_docker_env():
        lib_path = hass.config.path('deps')
        if os.path.isdir(lib_path):
            shutil.rmtree(lib_path)
    with open(version_path, 'w', encoding='utf8') as outp:
        outp.write(__version__)


@callback
def async_log_schema_error(exc: vol.Invalid, domain: str, config: Dict[str, Any], hass: HomeAssistant, link: Optional[str] = None) -> None:
    message: str = format_schema_error(hass, exc, domain, config, link)
    _LOGGER.error(message)


@callback
def async_log_config_validator_error(exc: Exception, domain: str, config: Dict[str, Any], hass: HomeAssistant, link: Optional[str] = None) -> None:
    if isinstance(exc, vol.Invalid):
        async_log_schema_error(exc, domain, config, hass, link)
        return
    message: str = format_homeassistant_error(hass, exc, domain, config, link)
    _LOGGER.error(message, exc_info=exc)


def _get_annotation(item: Any) -> Optional[Tuple[str, Any]]:
    if not hasattr(item, '__config_file__'):
        return None
    return (getattr(item, '__config_file__'), getattr(item, '__line__', '?'))


def _get_by_path(data: Any, items: Sequence[Any]) -> Any:
    try:
        return reduce(operator.getitem, items, data)
    except (KeyError, IndexError, TypeError):
        return None


def find_annotation(config: Any, path: Sequence[Any]) -> Optional[Tuple[str, Any]]:
    def find_annotation_for_key(item: Dict[Any, Any], path: Sequence[Any], tail: Any) -> Optional[Tuple[str, Any]]:
        for key in item:
            if key == tail:
                annotation = _get_annotation(key)
                if annotation:
                    return annotation
                break
        return None

    def find_annotation_rec(config: Any, path: List[Any], tail: Optional[Any]) -> Optional[Tuple[str, Any]]:
        item = _get_by_path(config, path)
        if isinstance(item, dict) and tail is not None:
            tail_annotation = find_annotation_for_key(item, path, tail)
            if tail_annotation:
                return tail_annotation
        if isinstance(item, (dict, list)) and path and (key_annotation := find_annotation_for_key(_get_by_path(config, path[:-1]), path[:-1], path[-1])):
            return key_annotation
        annotation = _get_annotation(item)
        if annotation:
            return annotation
        if not path:
            return None
        tail = path.pop()
        if (annotation := find_annotation_rec(config, path, tail)):
            return annotation
        return _get_annotation(item)
    return find_annotation_rec(config, list(path), None)


def _relpath(hass: HomeAssistant, path: str) -> str:
    return os.path.relpath(path, hass.config.config_dir)


def stringify_invalid(
    hass: HomeAssistant,
    exc: vol.Invalid,
    domain: str,
    config: Dict[str, Any],
    link: Optional[str],
    max_sub_error_length: int
) -> str:
    if '.' in domain:
        integration_domain, _, platform_domain = domain.partition('.')
        message_prefix = f"Invalid config for '{platform_domain}' from integration '{integration_domain}'"
    else:
        message_prefix = f"Invalid config for '{domain}'"
    if domain != HOMEASSISTANT_DOMAIN and link:
        message_suffix = f', please check the docs at {link}'
    else:
        message_suffix = ''
    annotation = find_annotation(config, exc.path)
    if annotation:
        message_prefix += f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
    path_str = '->'.join((str(m) for m in exc.path))
    if exc.error_message == 'extra keys not allowed':
        return f"{message_prefix}: '{exc.path[-1]}' is an invalid option for '{domain}', check: {path_str}{message_suffix}"
    if exc.error_message == 'required key not provided':
        return f"{message_prefix}: required key '{exc.path[-1]}' not provided{message_suffix}"
    output = Exception.__str__(exc)
    if (error_type := exc.error_type):
        output += ' for ' + error_type
    offending_item_summary = repr(_get_by_path(config, exc.path))
    if len(offending_item_summary) > max_sub_error_length:
        offending_item_summary = f'{offending_item_summary[:max_sub_error_length - 3]}...'
    return f"{message_prefix}: {output} '{path_str}', got {offending_item_summary}{message_suffix}"


def humanize_error(
    hass: HomeAssistant,
    validation_error: vol.Invalid,
    domain: str,
    config: Dict[str, Any],
    link: Optional[str],
    max_sub_error_length: int = MAX_VALIDATION_ERROR_ITEM_LENGTH
) -> str:
    if isinstance(validation_error, vol.MultipleInvalid):
        return '\n'.join(sorted(
            humanize_error(hass, sub_error, domain, config, link, max_sub_error_length)
            for sub_error in validation_error.errors
        ))
    return stringify_invalid(hass, validation_error, domain, config, link, max_sub_error_length)


@callback
def format_homeassistant_error(
    hass: HomeAssistant,
    exc: Exception,
    domain: str,
    config: Dict[str, Any],
    link: Optional[str] = None
) -> str:
    if '.' in domain:
        integration_domain, _, platform_domain = domain.partition('.')
        message_prefix = f"Invalid config for '{platform_domain}' from integration '{integration_domain}'"
    else:
        message_prefix = f"Invalid config for '{domain}'"
    annotation = find_annotation(config, [domain])
    if annotation:
        message_prefix += f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
    message = f'{message_prefix}: {str(exc) or repr(exc)}'
    if domain != HOMEASSISTANT_DOMAIN and link:
        message += f', please check the docs at {link}'
    return message


@callback
def format_schema_error(
    hass: HomeAssistant,
    exc: vol.Invalid,
    domain: str,
    config: Dict[str, Any],
    link: Optional[str] = None
) -> str:
    return humanize_error(hass, exc, domain, config, link)


def _log_pkg_error(
    hass: HomeAssistant,
    package: str,
    component: Optional[str],
    config: Dict[str, Any],
    message: str
) -> None:
    message_prefix = f"Setup of package '{package}'"
    annotation = find_annotation(config, [HOMEASSISTANT_DOMAIN, CONF_PACKAGES, package])
    if annotation:
        message_prefix += f' at {_relpath(hass, annotation[0])}, line {annotation[1]}'
    _LOGGER.error('%s failed: %s', message_prefix, message)


def _identify_config_schema(module: ModuleType) -> Optional[str]:
    if not isinstance(module.CONFIG_SCHEMA, vol.Schema):
        return None
    schema = module.CONFIG_SCHEMA.schema
    if isinstance(schema, vol.All):
        for subschema in schema.validators:
            if isinstance(subschema, dict):
                schema = subschema
                break
        else:
            return None
    try:
        key = next((k for k in schema if k == module.DOMAIN))
    except (TypeError, AttributeError, StopIteration):
        return None
    except Exception:
        _LOGGER.exception('Unexpected error identifying config schema')
        return None
    if hasattr(key, 'default') and (not isinstance(key.default, vol.schema_builder.Undefined)):
        default_value = module.CONFIG_SCHEMA({module.DOMAIN: key.default()})[module.DOMAIN]
        if isinstance(default_value, dict):
            return 'dict'
        if isinstance(default_value, list):
            return 'list'
        return None
    domain_schema = schema[key]
    t_schema = str(domain_schema)
    if t_schema.startswith('{') or 'schema_with_slug_keys' in t_schema:
        return 'dict'
    if t_schema.startswith(('[', 'All(<function ensure_list')):
        return 'list'
    return None


def _validate_package_definition(name: str, conf: Dict[str, Any]) -> None:
    cv.slug(name)
    _PACKAGE_DEFINITION_SCHEMA(conf)


def _recursive_merge(conf: Dict[str, Any], package: Dict[str, Any]) -> Optional[str]:
    duplicate_key: Optional[str] = None
    for key, pack_conf in package.items():
        if isinstance(pack_conf, dict):
            if not pack_conf:
                continue
            conf[key] = conf.get(key, OrderedDict())
            duplicate_key = _recursive_merge(conf=conf[key], package=pack_conf)
        elif isinstance(pack_conf, list):
            conf[key] = cv.remove_falsy(cv.ensure_list(conf.get(key)) + cv.ensure_list(pack_conf))
        else:
            if conf.get(key) is not None:
                return key
            conf[key] = pack_conf
    return duplicate_key


async def merge_packages_config(
    hass: HomeAssistant,
    config: Dict[str, Any],
    packages: Dict[str, Any],
    _log_pkg_error: Callable[[HomeAssistant, str, Optional[str], Dict[str, Any], str], None] = _log_pkg_error
) -> Dict[str, Any]:
    _PACKAGES_CONFIG_SCHEMA(packages)
    invalid_packages: List[str] = []
    for pack_name, pack_conf in packages.items():
        try:
            _validate_package_definition(pack_name, pack_conf)
        except vol.Invalid as exc:
            _log_pkg_error(hass, pack_name, None, config, f"Invalid package definition '{pack_name}': {exc!s}. Package will not be initialized")
            invalid_packages.append(pack_name)
            continue
        for comp_name, comp_conf in pack_conf.items():
            if comp_name == HOMEASSISTANT_DOMAIN:
                continue
            try:
                domain = cv.domain_key(comp_name)
            except vol.Invalid:
                _log_pkg_error(hass, pack_name, comp_name, config, f"Invalid domain '{comp_name}'")
                continue
            try:
                integration = await async_get_integration_with_requirements(hass, domain)
                component = await integration.async_get_component()
            except LOAD_EXCEPTIONS as exc:
                _log_pkg_error(hass, pack_name, comp_name, config, f'Integration {comp_name} caused error: {exc!s}')
                continue
            except INTEGRATION_LOAD_EXCEPTIONS as exc:
                _log_pkg_error(hass, pack_name, comp_name, config, str(exc))
                continue
            try:
                config_platform = await integration.async_get_platform('config')
                if not hasattr(config_platform, 'async_validate_config'):
                    config_platform = None
            except ImportError:
                config_platform = None
            merge_list: bool = False
            if config_platform is not None:
                merge_list = config_platform.PACKAGE_MERGE_HINT == 'list'
            if not merge_list:
                merge_list = hasattr(component, 'PLATFORM_SCHEMA')
            if not merge_list and hasattr(component, 'CONFIG_SCHEMA'):
                merge_list = _identify_config_schema(component) == 'list'
            if merge_list:
                config[comp_name] = cv.remove_falsy(cv.ensure_list(config.get(comp_name)) + cv.ensure_list(comp_conf))
                continue
            if comp_conf is None:
                comp_conf = OrderedDict()
            if not isinstance(comp_conf, dict):
                _log_pkg_error(hass, pack_name, comp_name, config, f"integration '{comp_name}' cannot be merged, expected a dict")
                continue
            if comp_name not in config or config[comp_name] is None:
                config[comp_name] = OrderedDict()
            if not isinstance(config[comp_name], dict):
                _log_pkg_error(hass, pack_name, comp_name, config, f"integration '{comp_name}' cannot be merged, dict expected in main config")
                continue
            duplicate_key = _recursive_merge(conf=config[comp_name], package=comp_conf)
            if duplicate_key:
                _log_pkg_error(hass, pack_name, comp_name, config, f"integration '{comp_name}' has duplicate key '{duplicate_key}'")
    for pack_name in invalid_packages:
        packages.pop(pack_name, {})
    return config


@dataclass(slots=True)
class _PlatformIntegration:
    platform_path: str
    p_name: str
    integration: Integration
    config: Any
    validated_config: Any


async def _async_load_and_validate_platform_integration(
    domain: str,
    integration_docs: Optional[str],
    config_exceptions: List[ConfigExceptionInfo],
    p_integration: _PlatformIntegration
) -> Optional[Any]:
    try:
        platform = await p_integration.integration.async_get_platform(domain)
    except LOAD_EXCEPTIONS as exc:
        exc_info = ConfigExceptionInfo(
            exc,
            ConfigErrorTranslationKey.PLATFORM_COMPONENT_LOAD_EXC,
            p_integration.platform_path,
            p_integration.config,
            integration_docs
        )
        config_exceptions.append(exc_info)
        return None
    if not hasattr(platform, 'PLATFORM_SCHEMA'):
        return p_integration.validated_config
    try:
        return platform.PLATFORM_SCHEMA(p_integration.config)
    except vol.Invalid as exc:
        exc_info = ConfigExceptionInfo(
            exc,
            ConfigErrorTranslationKey.PLATFORM_CONFIG_VALIDATION_ERR,
            p_integration.platform_path,
            p_integration.config,
            p_integration.integration.documentation
        )
        config_exceptions.append(exc_info)
    except Exception as exc:
        exc_info = ConfigExceptionInfo(
            exc,
            ConfigErrorTranslationKey.PLATFORM_SCHEMA_VALIDATOR_ERR,
            p_integration.p_name,
            p_integration.config,
            p_integration.integration.documentation
        )
        config_exceptions.append(exc_info)
    return None


async def async_process_component_and_handle_errors(
    hass: HomeAssistant,
    config: Dict[str, Any],
    integration: Integration,
    raise_on_failure: bool = False
) -> Optional[Dict[str, Any]]:
    integration_config_info: IntegrationConfigInfo = await async_process_component_config(hass, config, integration)
    async_handle_component_errors(hass, integration_config_info, integration, raise_on_failure)
    return async_drop_config_annotations(integration_config_info, integration)


@callback
def async_drop_config_annotations(
    integration_config_info: IntegrationConfigInfo,
    integration: Integration
) -> Optional[Dict[str, Any]]:
    config = integration_config_info.config
    if config is None:
        return None

    def drop_config_annotations_rec(node: Any) -> Any:
        if isinstance(node, dict):
            tmp = dict(node)
            node.clear()
            node.update({drop_config_annotations_rec(k): drop_config_annotations_rec(v) for k, v in tmp.items()})
            return node
        if isinstance(node, list):
            return [drop_config_annotations_rec(v) for v in node]
        if isinstance(node, NodeStrClass):
            return str(node)
        return node

    if integration.domain in config and integration.domain != HOMEASSISTANT_DOMAIN:
        drop_config_annotations_rec(config[integration.domain])
    return config


@callback
def async_handle_component_errors(
    hass: HomeAssistant,
    integration_config_info: IntegrationConfigInfo,
    integration: Integration,
    raise_on_failure: bool = False
) -> None:
    if not integration_config_info.exception_info_list:
        return
    domain: str = integration.domain
    for platform_exception in integration_config_info.exception_info_list:
        exception: Exception = platform_exception.exception
        log_message, show_stack_trace, placeholders = _get_log_message_and_stack_print_pref(hass, domain, platform_exception)
        _LOGGER.error(log_message, exc_info=exception if show_stack_trace else None)
    if not raise_on_failure:
        return
    if len(integration_config_info.exception_info_list) == 1:
        translation_key = integration_config_info.exception_info_list[0].translation_key
    else:
        translation_key = ConfigErrorTranslationKey.MULTIPLE_INTEGRATION_CONFIG_ERRORS
        errors = str(len(integration_config_info.exception_info_list))
        placeholders = {'domain': domain, 'errors': errors}
    raise ConfigValidationError(
        translation_key,
        [pe.exception for pe in integration_config_info.exception_info_list],
        translation_domain=HOMEASSISTANT_DOMAIN,
        translation_placeholders=placeholders
    )


def config_per_platform(config: Dict[str, Any], domain: str) -> Iterable[Tuple[Optional[str], Any]]:
    for config_key in extract_domain_configs(config, domain):
        platform_config = config[config_key]
        if not platform_config:
            continue
        if not isinstance(platform_config, list):
            platform_config = [platform_config]
        for item in platform_config:
            try:
                platform = item.get(CONF_PLATFORM)
            except AttributeError:
                platform = None
            yield (platform, item)


def extract_platform_integrations(config: Dict[str, Any], domains: Iterable[str]) -> Dict[str, set[Hashable]]:
    platform_integrations: Dict[str, set[Hashable]] = {}
    for key, domain_config in config.items():
        try:
            domain_key_val = cv.domain_key(key)
        except vol.Invalid:
            continue
        if domain_key_val not in domains:
            continue
        if not isinstance(domain_config, list):
            domain_config = [domain_config]
        for item in domain_config:
            try:
                platform = item.get(CONF_PLATFORM)
            except AttributeError:
                continue
            if platform and isinstance(platform, Hashable):
                platform_integrations.setdefault(domain_key_val, set()).add(platform)
    return platform_integrations


def extract_domain_configs(config: Dict[str, Any], domain: str) -> List[str]:
    domain_configs: List[str] = []
    for key in config:
        with suppress(vol.Invalid):
            if cv.domain_key(key) != domain:
                continue
            domain_configs.append(key)
    return domain_configs


async def _async_load_and_validate_platform_integration_wrapper(
    domain: str,
    integration_docs: Optional[str],
    config_exceptions: List[ConfigExceptionInfo],
    p_integration: _PlatformIntegration
) -> Optional[Any]:
    return await _async_load_and_validate_platform_integration(domain, integration_docs, config_exceptions, p_integration)


async def async_process_component_config(
    hass: HomeAssistant,
    config: Dict[str, Any],
    integration: Integration,
    component: Optional[Any] = None
) -> IntegrationConfigInfo:
    domain: str = integration.domain
    integration_docs: Optional[str] = integration.documentation
    config_exceptions: List[ConfigExceptionInfo] = []
    if not component:
        try:
            component = await integration.async_get_component()
        except LOAD_EXCEPTIONS as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.COMPONENT_IMPORT_ERR,
                domain,
                config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            return IntegrationConfigInfo(None, config_exceptions)
    config_validator: Optional[Any] = None
    if integration.platforms_exists(('config',)):
        try:
            config_validator = await integration.async_get_platform('config')
        except ImportError as err:
            if err.name != f'{integration.pkg_path}.config':
                exc_info = ConfigExceptionInfo(
                    err,
                    ConfigErrorTranslationKey.CONFIG_PLATFORM_IMPORT_ERR,
                    domain,
                    config,
                    integration_docs
                )
                config_exceptions.append(exc_info)
                return IntegrationConfigInfo(None, config_exceptions)
    if config_validator is not None and hasattr(config_validator, 'async_validate_config'):
        try:
            validated = await config_validator.async_validate_config(hass, config)
            return IntegrationConfigInfo(validated, [])
        except (vol.Invalid, HomeAssistantError) as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.CONFIG_VALIDATION_ERR,
                domain,
                config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            return IntegrationConfigInfo(None, config_exceptions)
        except Exception as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.CONFIG_VALIDATOR_UNKNOWN_ERR,
                domain,
                config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            return IntegrationConfigInfo(None, config_exceptions)
    if hasattr(component, 'CONFIG_SCHEMA'):
        try:
            validated = await cv.async_validate(hass, component.CONFIG_SCHEMA, config)
            return IntegrationConfigInfo(validated, [])
        except vol.Invalid as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.CONFIG_VALIDATION_ERR,
                domain,
                config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            return IntegrationConfigInfo(None, config_exceptions)
        except Exception as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.CONFIG_SCHEMA_UNKNOWN_ERR,
                domain,
                config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            return IntegrationConfigInfo(None, config_exceptions)
    component_platform_schema = getattr(component, 'PLATFORM_SCHEMA_BASE', getattr(component, 'PLATFORM_SCHEMA', None))
    if component_platform_schema is None:
        return IntegrationConfigInfo(config, [])
    platform_integrations_to_load: List[_PlatformIntegration] = []
    platforms: List[Any] = []
    for p_name, p_config in config_per_platform(config, domain):
        platform_path: str = f'{p_name}.{domain}'
        try:
            p_validated = await cv.async_validate(hass, component_platform_schema, p_config)
        except vol.Invalid as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.PLATFORM_CONFIG_VALIDATION_ERR,
                domain,
                p_config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            continue
        except Exception as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.PLATFORM_SCHEMA_VALIDATOR_ERR,
                str(p_name),
                config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            continue
        if p_name is None:
            platforms.append(p_validated)
            continue
        try:
            p_integration_inst = await async_get_integration_with_requirements(hass, p_name)
        except (RequirementsNotFound, IntegrationNotFound) as exc:
            exc_info = ConfigExceptionInfo(
                exc,
                ConfigErrorTranslationKey.PLATFORM_COMPONENT_LOAD_ERR,
                platform_path,
                p_config,
                integration_docs
            )
            config_exceptions.append(exc_info)
            continue
        p_integration_obj = _PlatformIntegration(platform_path, p_name, p_integration_inst, p_config, p_validated)
        platform_integrations_to_load.append(p_integration_obj)
    if platform_integrations_to_load:
        async_load_and_validate = partial(_async_load_and_validate_platform_integration, domain, integration_docs, config_exceptions)
        tasks = [
            create_eager_task(async_load_and_validate(p_inst), loop=hass.loop)
            for p_inst in platform_integrations_to_load
        ]
        loaded_results = await asyncio.gather(*tasks)
        platforms.extend([validated_config for validated_config in loaded_results if validated_config is not None])
    config = config_without_domain(config, domain)
    config[domain] = platforms
    return IntegrationConfigInfo(config, config_exceptions)


@callback
def config_without_domain(config: Dict[str, Any], domain: str) -> Dict[str, Any]:
    filter_keys = extract_domain_configs(config, domain)
    return {key: value for key, value in config.items() if key not in filter_keys}


async def async_check_ha_config_file(hass: HomeAssistant) -> Optional[str]:
    from .helpers import check_config
    res = await check_config.async_check_ha_config_file(hass)
    if not res.errors:
        return None
    return res.error_str


def safe_mode_enabled(config_dir: str) -> bool:
    safe_mode_path: str = os.path.join(config_dir, SAFE_MODE_FILENAME)
    safe_mode: bool = os.path.exists(safe_mode_path)
    if safe_mode:
        os.remove(safe_mode_path)
    return safe_mode


async def async_enable_safe_mode(hass: HomeAssistant) -> None:
    def _enable_safe_mode() -> None:
        Path(hass.config.path(SAFE_MODE_FILENAME)).touch()
    await hass.async_add_executor_job(_enable_safe_mode)


def _get_log_message_and_stack_print_pref(
    hass: HomeAssistant,
    domain: str,
    platform_exception: ConfigExceptionInfo
) -> Tuple[str, bool, Dict[str, Any]]:
    exception: Exception = platform_exception.exception
    platform_path: Any = platform_exception.path
    platform_config: Any = platform_exception.config
    link: Optional[str] = platform_exception.integration_docs
    placeholders: Dict[str, Any] = {
        'domain': domain,
        'error': str(exception),
        'p_name': platform_path,
        'config_file': '?',
        'line': '?'
    }
    show_stack_trace: Optional[bool] = _CONFIG_LOG_SHOW_STACK_TRACE.get(platform_exception.translation_key)
    if show_stack_trace is None:
        show_stack_trace = False
        if isinstance(exception, vol.Invalid):
            log_message = format_schema_error(hass, exception, platform_path, platform_config, link)
            annotation = find_annotation(platform_config, exception.path)
            if annotation:
                placeholders['config_file'], line = annotation
                placeholders['line'] = str(line)
        else:
            log_message = format_homeassistant_error(hass, exception, platform_path, platform_config, link)
            annotation = find_annotation(platform_config, [platform_path])
            if annotation:
                placeholders['config_file'], line = annotation
                placeholders['line'] = str(line)
            show_stack_trace = True
        return (log_message, show_stack_trace, placeholders)
    log_message = async_get_exception_message(HOMEASSISTANT_DOMAIN, platform_exception.translation_key, translation_placeholders=placeholders)
    return (log_message, show_stack_trace, placeholders)