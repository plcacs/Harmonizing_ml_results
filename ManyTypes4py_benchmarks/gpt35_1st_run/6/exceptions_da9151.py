from __future__ import annotations
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Dict, Union

if TYPE_CHECKING:
    from .core import Context

_function_cache: Dict[str, Callable] = {}

def import_async_get_exception_message() -> Callable:
    ...

class HomeAssistantError(Exception):
    _message: str = None
    generate_message: bool = False

    def __init__(self, *args, translation_domain: str = None, translation_key: str = None, translation_placeholders: Dict[str, Any] = None):
        ...

    def __str__(self) -> str:
        ...

class ConfigValidationError(HomeAssistantError, ExceptionGroup[Exception]):
    ...

class ServiceValidationError(HomeAssistantError):
    ...

class InvalidEntityFormatError(HomeAssistantError):
    ...

class NoEntitySpecifiedError(HomeAssistantError):
    ...

class TemplateError(HomeAssistantError):
    ...

@dataclass(slots=True)
class ConditionError(HomeAssistantError):
    ...

@dataclass(slots=True)
class ConditionErrorMessage(ConditionError):
    ...

@dataclass(slots=True)
class ConditionErrorIndex(ConditionError):
    ...

@dataclass(slots=True)
class ConditionErrorContainer(ConditionError):
    ...

class IntegrationError(HomeAssistantError):
    ...

class PlatformNotReady(IntegrationError):
    ...

class ConfigEntryError(IntegrationError):
    ...

class ConfigEntryNotReady(IntegrationError):
    ...

class ConfigEntryAuthFailed(IntegrationError):
    ...

class InvalidStateError(HomeAssistantError):
    ...

class Unauthorized(HomeAssistantError):
    ...

class UnknownUser(Unauthorized):
    ...

class ServiceNotFound(ServiceValidationError):
    ...

class ServiceNotSupported(ServiceValidationError):
    ...

class MaxLengthExceeded(HomeAssistantError):
    ...

class DependencyError(HomeAssistantError):
    ...
