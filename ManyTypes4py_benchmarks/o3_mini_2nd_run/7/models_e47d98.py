from __future__ import annotations
import asyncio
import logging
import pathlib
import shutil
from collections.abc import Awaitable, Callable
from typing import Any, Dict, Optional, Union

import voluptuous as vol
from voluptuous.humanize import humanize_error

from homeassistant import loader
from homeassistant.const import CONF_DEFAULT, CONF_DOMAIN, CONF_NAME, CONF_PATH, __version__
from homeassistant.core import DOMAIN as HOMEASSISTANT_DOMAIN, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import yaml as yaml_util

from awesomeversion import AwesomeVersion

from .const import BLUEPRINT_FOLDER, CONF_BLUEPRINT, CONF_HOMEASSISTANT, CONF_INPUT, CONF_MIN_VERSION, CONF_SOURCE_URL, CONF_USE_BLUEPRINT, DOMAIN
from .errors import BlueprintException, BlueprintInUse, FailedToLoad, FileAlreadyExists, InvalidBlueprint, InvalidBlueprintInputs, MissingInput
from .schemas import BLUEPRINT_INSTANCE_FIELDS

BlueprintData = Dict[str, Any]
BlueprintSchema = Callable[[Any], BlueprintData]


class Blueprint:
    """Blueprint of a configuration structure."""

    def __init__(
        self,
        data: Any,
        *,
        path: Optional[str] = None,
        expected_domain: Optional[str] = None,
        schema: BlueprintSchema,
    ) -> None:
        """Initialize a blueprint."""
        try:
            data = self.data = schema(data)
        except vol.Invalid as err:
            raise InvalidBlueprint(expected_domain, path, data, err) from err
        data_domain: str = data[CONF_BLUEPRINT][CONF_DOMAIN]
        if expected_domain is not None and data_domain != expected_domain:
            raise InvalidBlueprint(
                expected_domain,
                path or self.name,
                data,
                f"Found incorrect blueprint type {data_domain}, expected {expected_domain}",
            )
        self.domain: str = data_domain
        missing = yaml_util.extract_inputs(data) - set(self.inputs)
        if missing:
            raise InvalidBlueprint(
                data_domain,
                path or self.name,
                data,
                f"Missing input definition for {', '.join(missing)}",
            )

    @property
    def name(self) -> str:
        """Return blueprint name."""
        return self.data[CONF_BLUEPRINT][CONF_NAME]

    @property
    def inputs(self) -> Dict[str, Any]:
        """Return flattened blueprint inputs."""
        inputs: Dict[str, Any] = {}
        for key, value in self.data[CONF_BLUEPRINT][CONF_INPUT].items():
            if value and CONF_INPUT in value:
                inputs.update(dict(value[CONF_INPUT]))
            else:
                inputs[key] = value
        return inputs

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return blueprint metadata."""
        return self.data[CONF_BLUEPRINT]

    def update_metadata(self, *, source_url: Optional[str] = None) -> None:
        """Update metadata."""
        if source_url is not None:
            self.data[CONF_BLUEPRINT][CONF_SOURCE_URL] = source_url

    def yaml(self) -> str:
        """Dump blueprint as YAML."""
        return yaml_util.dump(self.data)

    @callback
    def validate(self) -> Optional[list[str]]:
        """Test if the Home Assistant installation supports this blueprint.

        Return list of errors if not valid.
        """
        errors: list[str] = []
        metadata = self.metadata
        min_version = metadata.get(CONF_HOMEASSISTANT, {}).get(CONF_MIN_VERSION)
        if min_version is not None and AwesomeVersion(__version__) < AwesomeVersion(min_version):
            errors.append(f"Requires at least Home Assistant {min_version}")
        return errors or None


class BlueprintInputs:
    """Inputs for a blueprint."""

    def __init__(self, blueprint: Blueprint, config_with_inputs: Dict[str, Any]) -> None:
        """Instantiate a blueprint inputs object."""
        self.blueprint: Blueprint = blueprint
        self.config_with_inputs: Dict[str, Any] = config_with_inputs

    @property
    def inputs(self) -> Dict[str, Any]:
        """Return the inputs."""
        return self.config_with_inputs[CONF_USE_BLUEPRINT][CONF_INPUT]

    @property
    def inputs_with_default(self) -> Dict[str, Any]:
        """Return the inputs and fallback to defaults."""
        no_input = set(self.blueprint.inputs) - set(self.inputs)
        inputs_with_default: Dict[str, Any] = dict(self.inputs)
        for inp in no_input:
            blueprint_input = self.blueprint.inputs[inp]
            if isinstance(blueprint_input, dict) and CONF_DEFAULT in blueprint_input:
                inputs_with_default[inp] = blueprint_input[CONF_DEFAULT]
        return inputs_with_default

    def validate(self) -> None:
        """Validate the inputs."""
        missing = set(self.blueprint.inputs) - set(self.inputs_with_default)
        if missing:
            raise MissingInput(self.blueprint.domain, self.blueprint.name, missing)

    @callback
    def async_substitute(self) -> Dict[str, Any]:
        """Get the blueprint value with the inputs substituted."""
        processed = yaml_util.substitute(self.blueprint.data, self.inputs_with_default)
        combined: Dict[str, Any] = {**processed, **self.config_with_inputs}
        combined.pop(CONF_USE_BLUEPRINT)
        combined.pop(CONF_BLUEPRINT)
        return combined


class DomainBlueprints:
    """Blueprints for a specific domain."""

    def __init__(
        self,
        hass: HomeAssistant,
        domain: str,
        logger: logging.Logger,
        blueprint_in_use: Callable[[HomeAssistant, str], bool],
        reload_blueprint_consumers: Callable[[HomeAssistant, str], Awaitable[Any]],
        blueprint_schema: BlueprintSchema,
    ) -> None:
        """Initialize a domain blueprints instance."""
        self.hass: HomeAssistant = hass
        self.domain: str = domain
        self.logger: logging.Logger = logger
        self._blueprint_in_use: Callable[[HomeAssistant, str], bool] = blueprint_in_use
        self._reload_blueprint_consumers: Callable[[HomeAssistant, str], Awaitable[Any]] = reload_blueprint_consumers
        self._blueprints: Dict[str, Optional[Blueprint]] = {}
        self._load_lock: asyncio.Lock = asyncio.Lock()
        self._blueprint_schema: BlueprintSchema = blueprint_schema
        hass.data.setdefault(DOMAIN, {})[domain] = self

    @property
    def blueprint_folder(self) -> pathlib.Path:
        """Return the blueprint folder."""
        return pathlib.Path(self.hass.config.path(BLUEPRINT_FOLDER, self.domain))

    async def async_reset_cache(self) -> None:
        """Reset the blueprint cache."""
        async with self._load_lock:
            self._blueprints = {}

    def _load_blueprint(self, blueprint_path: str) -> Blueprint:
        """Load a blueprint."""
        try:
            blueprint_data: Dict[str, Any] = yaml_util.load_yaml_dict(self.blueprint_folder / blueprint_path)
        except FileNotFoundError as err:
            raise FailedToLoad(self.domain, blueprint_path, FileNotFoundError(f"Unable to find {blueprint_path}")) from err
        except HomeAssistantError as err:
            raise FailedToLoad(self.domain, blueprint_path, err) from err
        return Blueprint(blueprint_data, expected_domain=self.domain, path=blueprint_path, schema=self._blueprint_schema)

    def _load_blueprints(self) -> Dict[str, Union[Blueprint, Exception]]:
        """Load all the blueprints."""
        blueprint_folder = pathlib.Path(self.hass.config.path(BLUEPRINT_FOLDER, self.domain))
        results: Dict[str, Union[Blueprint, Exception]] = {}
        for path in blueprint_folder.glob("**/*.yaml"):
            blueprint_path = str(path.relative_to(blueprint_folder))
            if self._blueprints.get(blueprint_path) is None:
                try:
                    self._blueprints[blueprint_path] = self._load_blueprint(blueprint_path)
                except BlueprintException as err:
                    self._blueprints[blueprint_path] = None
                    results[blueprint_path] = err
                    continue
            results[blueprint_path] = self._blueprints[blueprint_path]  # type: ignore
        return results

    async def async_get_blueprints(self) -> Dict[str, Union[Blueprint, Exception]]:
        """Get all the blueprints."""
        async with self._load_lock:
            return await self.hass.async_add_executor_job(self._load_blueprints)

    async def async_get_blueprint(self, blueprint_path: str) -> Blueprint:
        """Get a blueprint."""

        def load_from_cache() -> Blueprint:
            """Load blueprint from cache."""
            blueprint = self._blueprints[blueprint_path]
            if blueprint is None:
                raise FailedToLoad(self.domain, blueprint_path, FileNotFoundError(f"Unable to find {blueprint_path}"))
            return blueprint

        if blueprint_path in self._blueprints:
            return load_from_cache()
        async with self._load_lock:
            if blueprint_path in self._blueprints:
                return load_from_cache()
            try:
                blueprint = await self.hass.async_add_executor_job(self._load_blueprint, blueprint_path)
            except FailedToLoad:
                self._blueprints[blueprint_path] = None
                raise
            self._blueprints[blueprint_path] = blueprint
            return blueprint

    async def async_inputs_from_config(
        self, config_with_blueprint: Dict[str, Any]
    ) -> BlueprintInputs:
        """Process a blueprint config."""
        try:
            config_with_blueprint = BLUEPRINT_INSTANCE_FIELDS(config_with_blueprint)
        except vol.Invalid as err:
            raise InvalidBlueprintInputs(self.domain, humanize_error(config_with_blueprint, err)) from err
        bp_conf: Dict[str, Any] = config_with_blueprint[CONF_USE_BLUEPRINT]
        blueprint = await self.async_get_blueprint(bp_conf[CONF_PATH])
        inputs = BlueprintInputs(blueprint, config_with_blueprint)
        inputs.validate()
        return inputs

    async def async_remove_blueprint(self, blueprint_path: str) -> None:
        """Remove a blueprint file."""
        if self._blueprint_in_use(self.hass, blueprint_path):
            raise BlueprintInUse(self.domain, blueprint_path)
        path = self.blueprint_folder / blueprint_path
        await self.hass.async_add_executor_job(path.unlink)
        self._blueprints[blueprint_path] = None

    def _create_file(self, blueprint: Blueprint, blueprint_path: str, allow_override: bool) -> bool:
        """Create blueprint file.

        Returns true if the action overrides an existing blueprint.
        """
        path = pathlib.Path(self.hass.config.path(BLUEPRINT_FOLDER, self.domain, blueprint_path))
        exists = path.exists()
        if not allow_override and exists:
            raise FileAlreadyExists(self.domain, blueprint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(blueprint.yaml(), encoding="utf-8")
        return exists

    async def async_add_blueprint(
        self, blueprint: Blueprint, blueprint_path: str, allow_override: bool = False
    ) -> bool:
        """Add a blueprint."""
        overrides_existing = await self.hass.async_add_executor_job(
            self._create_file, blueprint, blueprint_path, allow_override
        )
        self._blueprints[blueprint_path] = blueprint
        if overrides_existing:
            await self._reload_blueprint_consumers(self.hass, blueprint_path)
        return overrides_existing

    async def async_populate(self) -> None:
        """Create folder if it doesn't exist and populate with examples."""
        if self._blueprints:
            return
        integration = await loader.async_get_integration(self.hass, self.domain)

        def populate() -> None:
            if self.blueprint_folder.exists():
                return
            shutil.copytree(integration.file_path / BLUEPRINT_FOLDER, self.blueprint_folder / HOMEASSISTANT_DOMAIN)

        await self.hass.async_add_executor_job(populate)