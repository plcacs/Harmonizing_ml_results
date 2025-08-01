from __future__ import annotations
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, Dict

from .util.event_type import EventType

if TYPE_CHECKING:
    from .core import Context

_function_cache: Dict[str, Callable[[str, str, Optional[Dict[str, Any]]], str]] = {}

def import_async_get_exception_message() -> Callable[[str, str, Optional[Dict[str, Any]]], str]:
    """Return a method that can fetch a translated exception message.

    Defaults to English, requires translations to already be cached.
    """
    from .helpers.translation import async_get_exception_message as async_get_exception_message_import
    return async_get_exception_message_import

class HomeAssistantError(Exception):
    """General Home Assistant exception occurred."""
    _message: Optional[str] = None
    generate_message: bool = False

    def __init__(
        self,
        *args: Any,
        translation_domain: Optional[str] = None,
        translation_key: Optional[str] = None,
        translation_placeholders: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize exception."""
        if not args and translation_key and translation_domain:
            self.generate_message = True
            args = (translation_key,)
        super().__init__(*args)
        self.translation_domain: Optional[str] = translation_domain
        self.translation_key: Optional[str] = translation_key
        self.translation_placeholders: Optional[Dict[str, Any]] = translation_placeholders

    def __str__(self) -> str:
        """Return exception message.

        If no message was passed to `__init__`, the exception message is generated from
        the translation_key. The message will be in English, regardless of the configured
        language.
        """
        if self._message:
            return self._message
        if not self.generate_message:
            self._message = super().__str__()
            return self._message
        if TYPE_CHECKING:
            assert self.translation_key is not None
            assert self.translation_domain is not None
        if "async_get_exception_message" not in _function_cache:
            _function_cache["async_get_exception_message"] = import_async_get_exception_message()
        self._message = _function_cache["async_get_exception_message"](
            self.translation_domain, self.translation_key, self.translation_placeholders
        )
        return self._message

class ConfigValidationError(HomeAssistantError, Exception):
    """A validation exception occurred when validating the configuration."""

    def __init__(
        self,
        message_translation_key: str,
        exceptions: Any,
        translation_domain: Optional[str] = None,
        translation_placeholders: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize exception."""
        super().__init__(
            message_translation_key,
            exceptions,
            translation_domain=translation_domain,
            translation_key=message_translation_key,
            translation_placeholders=translation_placeholders,
        )
        self.generate_message = True

class ServiceValidationError(HomeAssistantError):
    """A validation exception occurred when calling a service."""

class InvalidEntityFormatError(HomeAssistantError):
    """When an invalid formatted entity is encountered."""

class NoEntitySpecifiedError(HomeAssistantError):
    """When no entity is specified."""

class TemplateError(HomeAssistantError):
    """Error during template rendering."""

    def __init__(self, exception: Union[Exception, str]) -> None:
        """Init the error."""
        if isinstance(exception, str):
            super().__init__(exception)
        else:
            super().__init__(f"{exception.__class__.__name__}: {exception}")

@dataclass(slots=True)
class ConditionError(HomeAssistantError):
    """Error during condition evaluation."""

    def output(self, indent: int) -> Generator[str, None, None]:
        """Yield an indented representation."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return string representation."""
        return "\n".join(list(self.output(indent=0)))

@dataclass(slots=True)
class ConditionErrorMessage(ConditionError):
    """Condition error message."""
    type: str
    message: str

    def output(self, indent: int) -> Generator[str, None, None]:
        """Yield an indented representation."""
        yield self._indent(indent, f"In '{self.type}' condition: {self.message}")

@dataclass(slots=True)
class ConditionErrorIndex(ConditionError):
    """Condition error with index."""
    type: str
    index: int
    total: int
    error: ConditionError

    def output(self, indent: int) -> Generator[str, None, None]:
        """Yield an indented representation."""
        if self.total > 1:
            yield self._indent(indent, f"In '{self.type}' (item {self.index + 1} of {self.total}):")
        else:
            yield self._indent(indent, f"In '{self.type}':")
        yield from self.error.output(indent + 1)

@dataclass(slots=True)
class ConditionErrorContainer(ConditionError):
    """Condition error with subconditions."""
    errors: Sequence[ConditionError]

    def output(self, indent: int) -> Generator[str, None, None]:
        """Yield an indented representation."""
        for item in self.errors:
            yield from item.output(indent)

    # Providing a static indent helper for the container as well
    @staticmethod
    def _indent(indent: int, message: str) -> str:
        return "  " * indent + message

class IntegrationError(HomeAssistantError):
    """Base class for platform and config entry exceptions."""

    def __str__(self) -> str:
        """Return a human readable error."""
        return super().__str__() or str(self.__cause__)

class PlatformNotReady(IntegrationError):
    """Error to indicate that platform is not ready."""

class ConfigEntryError(IntegrationError):
    """Error to indicate that config entry setup has failed."""

class ConfigEntryNotReady(IntegrationError):
    """Error to indicate that config entry is not ready."""

class ConfigEntryAuthFailed(IntegrationError):
    """Error to indicate that config entry could not authenticate."""

class InvalidStateError(HomeAssistantError):
    """When an invalid state is encountered."""

class Unauthorized(HomeAssistantError):
    """When an action is unauthorized."""

    def __init__(
        self,
        context: Optional[Context] = None,
        user_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        config_entry_id: Optional[str] = None,
        perm_category: Optional[str] = None,
        permission: Optional[str] = None,
    ) -> None:
        """Unauthorized error."""
        super().__init__(self.__class__.__name__)
        self.context: Optional[Context] = context
        if user_id is None and context is not None:
            user_id = context.user_id
        self.user_id: Optional[str] = user_id
        self.entity_id: Optional[str] = entity_id
        self.config_entry_id: Optional[str] = config_entry_id
        self.perm_category: Optional[str] = perm_category
        self.permission: Optional[str] = permission

class UnknownUser(Unauthorized):
    """When call is made with user ID that doesn't exist."""

class ServiceNotFound(ServiceValidationError):
    """Raised when a service is not found."""

    def __init__(self, domain: str, service: str) -> None:
        """Initialize error."""
        super().__init__(
            translation_domain="homeassistant",
            translation_key="service_not_found",
            translation_placeholders={"domain": domain, "service": service},
        )
        self.domain: str = domain
        self.service: str = service
        self.generate_message = True

class ServiceNotSupported(ServiceValidationError):
    """Raised when an entity action is not supported."""

    def __init__(self, domain: str, service: str, entity_id: str) -> None:
        """Initialize ServiceNotSupported exception."""
        super().__init__(
            translation_domain="homeassistant",
            translation_key="service_not_supported",
            translation_placeholders={"domain": domain, "service": service, "entity_id": entity_id},
        )
        self.domain: str = domain
        self.service: str = service
        self.generate_message = True

class MaxLengthExceeded(HomeAssistantError):
    """Raised when a property value has exceeded the max character length."""

    def __init__(self, value: Any, property_name: str, max_length: int) -> None:
        """Initialize error."""
        if TYPE_CHECKING:
            value = str(value)
        super().__init__(
            translation_domain="homeassistant",
            translation_key="max_length_exceeded",
            translation_placeholders={"value": value, "property_name": property_name, "max_length": str(max_length)},
        )
        self.value: Any = value
        self.property_name: str = property_name
        self.max_length: int = max_length
        self.generate_message = True

class DependencyError(HomeAssistantError):
    """Raised when dependencies cannot be setup."""

    def __init__(self, failed_dependencies: Sequence[str]) -> None:
        """Initialize error."""
        super().__init__(f"Could not setup dependencies: {', '.join(failed_dependencies)}")
        self.failed_dependencies: Sequence[str] = failed_dependencies

# Add the _indent static method to HomeAssistantError for reused functionality
def _indent(indent: int, message: str) -> str:
    return "  " * indent + message

# Monkey-patch the _indent method into ConditionError to be used by its subclasses
ConditionError._indent = staticmethod(_indent)