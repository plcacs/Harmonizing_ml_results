from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import voluptuous as vol
from voluptuous.humanize import humanize_error
from homeassistant.exceptions import HomeAssistantError

class BlueprintException(HomeAssistantError):
    def __init__(self, domain: str, msg: str) -> None:
        super().__init__(msg)
        self.domain: str = domain

class BlueprintWithNameException(BlueprintException):
    def __init__(self, domain: str, blueprint_name: str, msg: str) -> None:
        super().__init__(domain, msg)
        self.blueprint_name: str = blueprint_name

class FailedToLoad(BlueprintWithNameException):
    def __init__(self, domain: str, blueprint_name: str, exc: Exception) -> None:
        super().__init__(domain, blueprint_name, f'Failed to load blueprint: {exc}')

class InvalidBlueprint(BlueprintWithNameException):
    def __init__(self, domain: str, blueprint_name: str, blueprint_data: Any, msg_or_exc: Any) -> None:
        if isinstance(msg_or_exc, vol.Invalid):
            msg_or_exc = humanize_error(blueprint_data, msg_or_exc)
        super().__init__(domain, blueprint_name, f'Invalid blueprint: {msg_or_exc}')
        self.blueprint_data: Any = blueprint_data

class InvalidBlueprintInputs(BlueprintException):
    def __init__(self, domain: str, msg: str) -> None:
        super().__init__(domain, f'Invalid blueprint inputs: {msg}')

class MissingInput(BlueprintWithNameException):
    def __init__(self, domain: str, blueprint_name: str, input_names: Iterable[str]) -> None:
        super().__init__(domain, blueprint_name, f'Missing input {', '.join(sorted(input_names))}')

class FileAlreadyExists(BlueprintWithNameException):
    def __init__(self, domain: str, blueprint_name: str) -> None:
        super().__init__(domain, blueprint_name, 'Blueprint already exists')

class BlueprintInUse(BlueprintWithNameException):
    def __init__(self, domain: str, blueprint_name: str) -> None:
        super().__init__(domain, blueprint_name, 'Blueprint in use')
