"""Helper to check the configuration file."""
from __future__ import annotations
from collections import OrderedDict
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Self, Set
import voluptuous as vol
from homeassistant import loader
from homeassistant.config import (
    CONF_PACKAGES,
    YAML_CONFIG_FILE,
    config_per_platform,
    extract_domain_configs,
    format_homeassistant_error,
    format_schema_error,
    load_yaml_config_file,
    merge_packages_config,
)
from homeassistant.core import DOMAIN as HOMEASSISTANT_DOMAIN, HomeAssistant
from homeassistant.core_config import CORE_CONFIG_SCHEMA
from homeassistant.exceptions import HomeAssistantError
from homeassistant.requirements import (
    RequirementsNotFound,
    async_clear_install_history,
    async_get_integration_with_requirements,
)
from homeassistant.util.yaml import loader as yaml_loader
from . import config_validation as cv
from .typing import ConfigType


class CheckConfigError(NamedTuple):
    """Configuration check error."""
    message: str
    domain: Optional[str]
    config: Any


class HomeAssistantConfig(OrderedDict[str, Any]):
    """Configuration result with errors attribute."""

    def __init__(self) -> None:
        """Initialize HA config."""
        super().__init__()
        self.errors: List[CheckConfigError] = []
        self.warnings: List[CheckConfigError] = []

    def add_error(self, message: Any, domain: Optional[str] = None, config: Any = None) -> Self:
        """Add an error."""
        self.errors.append(CheckConfigError(str(message), domain, config))
        return self

    @property
    def error_str(self) -> str:
        """Concatenate all errors to a string."""
        return '\n'.join([err.message for err in self.errors])

    def add_warning(self, message: Any, domain: Optional[str] = None, config: Any = None) -> Self:
        """Add a warning."""
        self.warnings.append(CheckConfigError(str(message), domain, config))
        return self

    @property
    def warning_str(self) -> str:
        """Concatenate all warnings to a string."""
        return '\n'.join([err.message for err in self.warnings])


async def async_check_ha_config_file(hass: HomeAssistant) -> HomeAssistantConfig:
    """Load and check if Home Assistant configuration file is valid.

    This method is a coroutine.
    """
    result: HomeAssistantConfig = HomeAssistantConfig()
    async_clear_install_history(hass)

    def _pack_error(
        hass: HomeAssistant,
        package: str,
        component: Optional[str],
        config: Any,
        message: str,
    ) -> None:
        """Handle errors from packages."""
        message = f"Setup of package '{package}' failed: {message}"
        domain = f"homeassistant.packages.{package}{('.' + component) if component is not None else ''}"
        pack_config = core_config[CONF_PACKAGES].get(package, config)
        result.add_warning(message, domain, pack_config)

    def _comp_error(
        ex: Exception,
        domain: str,
        component_config: Any,
        config_to_attach: Any,
    ) -> None:
        """Handle errors from components."""
        if isinstance(ex, vol.Invalid):
            message = format_schema_error(hass, ex, domain, component_config)
        else:
            message = format_homeassistant_error(hass, ex, domain, component_config)
        if domain in frontend_dependencies:
            result.add_error(message, domain, config_to_attach)
        else:
            result.add_warning(message, domain, config_to_attach)

    async def _get_integration(hass: HomeAssistant, domain: str) -> Optional[loader.Integration]:
        """Get an integration."""
        integration: Optional[loader.Integration] = None
        try:
            integration = await async_get_integration_with_requirements(hass, domain)
        except loader.IntegrationNotFound as ex:
            if not hass.config.recovery_mode and (not hass.config.safe_mode):
                result.add_warning(f'Integration error: {domain} - {ex}')
        except RequirementsNotFound as ex:
            result.add_warning(f'Integration error: {domain} - {ex}')
        return integration

    config_path: str = hass.config.path(YAML_CONFIG_FILE)
    try:
        if not await hass.async_add_executor_job(os.path.isfile, config_path):
            return result.add_error('File configuration.yaml not found.')
        config: ConfigType = await hass.async_add_executor_job(
            load_yaml_config_file, config_path, yaml_loader.Secrets(Path(hass.config.config_dir))
        )
    except FileNotFoundError:
        return result.add_error(f'File not found: {config_path}')
    except HomeAssistantError as err:
        return result.add_error(f'Error loading {config_path}: {err}')

    core_config: Dict[str, Any] = config.pop(HOMEASSISTANT_DOMAIN, {})
    try:
        core_config = CORE_CONFIG_SCHEMA(core_config)
        result[HOMEASSISTANT_DOMAIN] = core_config
        await merge_packages_config(
            hass, config, core_config.get(CONF_PACKAGES, {}), _pack_error
        )
    except vol.Invalid as err:
        result.add_error(
            format_schema_error(hass, err, HOMEASSISTANT_DOMAIN, core_config),
            HOMEASSISTANT_DOMAIN,
            core_config,
        )
        core_config = {}

    core_config.pop(CONF_PACKAGES, None)
    components: Set[str] = {cv.domain_key(key) for key in config}
    frontend_dependencies: Set[str] = set()
    if 'frontend' in components or 'default_config' in components:
        frontend = await _get_integration(hass, 'frontend')
        if frontend:
            await frontend.resolve_dependencies()
            frontend_dependencies = frontend.all_dependencies | {'frontend'}

    for domain in components:
        integration = await _get_integration(hass, domain)
        if not integration:
            continue

        try:
            component: Any = await integration.async_get_component()
        except ImportError as ex:
            result.add_warning(f'Component error: {domain} - {ex}')
            continue

        config_validator: Any = None
        if integration.platforms_exists(('config',)):
            try:
                config_validator = await integration.async_get_platform('config')
            except ImportError as err:
                if err.name != f'{integration.pkg_path}.config':
                    result.add_error(f'Error importing config platform {domain}: {err}')
                    continue

        if config_validator is not None and hasattr(config_validator, 'async_validate_config'):
            try:
                validated = await config_validator.async_validate_config(hass, config)
                result[domain] = validated[domain]
                continue
            except (vol.Invalid, HomeAssistantError) as ex:
                _comp_error(ex, domain, config, config[domain])
                continue
            except Exception as err:
                logging.getLogger(__name__).exception('Unexpected error validating config')
                result.add_error(
                    f'Unexpected error calling config validator: {err}',
                    domain,
                    config.get(domain),
                )
                continue

        config_schema: Any = getattr(component, 'CONFIG_SCHEMA', None)
        if config_schema is not None:
            try:
                validated_config: Dict[str, Any] = await cv.async_validate(hass, config_schema, config)
                if domain in validated_config:
                    result[domain] = validated_config[domain]
            except vol.Invalid as ex:
                _comp_error(ex, domain, config, config[domain])
                continue

        component_platform_schema: Any = getattr(
            component, 'PLATFORM_SCHEMA_BASE', getattr(component, 'PLATFORM_SCHEMA', None)
        )
        if component_platform_schema is None:
            continue

        platforms: List[Any] = []
        for p_name, p_config in config_per_platform(config, domain):
            try:
                p_validated: Any = await cv.async_validate(hass, component_platform_schema, p_config)
            except vol.Invalid as ex:
                _comp_error(ex, domain, p_config, p_config)
                continue

            if p_name is None:
                platforms.append(p_validated)
                continue

            try:
                p_integration: loader.Integration = await async_get_integration_with_requirements(hass, p_name)
                platform: Any = await p_integration.async_get_platform(domain)
            except loader.IntegrationNotFound as ex:
                if not hass.config.recovery_mode and (not hass.config.safe_mode):
                    result.add_warning(f"Platform error '{domain}' from integration '{p_name}' - {ex}")
                continue
            except (RequirementsNotFound, ImportError) as ex:
                result.add_warning(f"Platform error '{domain}' from integration '{p_name}' - {ex}")
                continue

            platform_schema: Any = getattr(platform, 'PLATFORM_SCHEMA', None)
            if platform_schema is not None:
                try:
                    p_validated = platform_schema(p_validated)
                except vol.Invalid as ex:
                    _comp_error(ex, f'{domain}.{p_name}', p_config, p_config)
                    continue
            platforms.append(p_validated)

        for filter_comp in extract_domain_configs(config, domain):
            del config[filter_comp]
        result[domain] = platforms

    return result