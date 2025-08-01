"""Pydantic-specific warnings."""
from __future__ import annotations as _annotations
from typing import Optional, Tuple, Any
from .version import version_short
__all__ = (
    'PydanticDeprecatedSince20',
    'PydanticDeprecatedSince26',
    'PydanticDeprecatedSince29',
    'PydanticDeprecatedSince210',
    'PydanticDeprecatedSince211',
    'PydanticDeprecationWarning',
    'PydanticExperimentalWarning',
)

class PydanticDeprecationWarning(DeprecationWarning):
    """A Pydantic specific deprecation warning.

    This warning is raised when using deprecated functionality in Pydantic. It provides information on when the
    deprecation was introduced and the expected version in which the corresponding functionality will be removed.

    Attributes:
        message: Description of the warning.
        since: Pydantic version in which the deprecation was introduced.
        expected_removal: Pydantic version in which the corresponding functionality is expected to be removed.
    """

    def __init__(self, message: str, *args: Any, *, since: Tuple[int, int], expected_removal: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(message, *args)
        self.message: str = message.rstrip('.')
        self.since: Tuple[int, int] = since
        self.expected_removal: Tuple[int, int] = expected_removal if expected_removal is not None else (since[0] + 1, 0)

    def __str__(self) -> str:
        message = f'{self.message}. Deprecated in Pydantic V{self.since[0]}.{self.since[1]} to be removed in V{self.expected_removal[0]}.{self.expected_removal[1]}.'
        if self.since == (2, 0):
            message += f' See Pydantic V2 Migration Guide at https://errors.pydantic.dev/{version_short()}/migration/'
        return message

class PydanticDeprecatedSince20(PydanticDeprecationWarning):
    """A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.0."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 0), expected_removal=(3, 0))

class PydanticDeprecatedSince26(PydanticDeprecationWarning):
    """A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.6."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 6), expected_removal=(3, 0))

class PydanticDeprecatedSince29(PydanticDeprecationWarning):
    """A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.9."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 9), expected_removal=(3, 0))

class PydanticDeprecatedSince210(PydanticDeprecationWarning):
    """A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.10."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 10), expected_removal=(3, 0))

class PydanticDeprecatedSince211(PydanticDeprecationWarning):
    """A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.11."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 11), expected_removal=(3, 0))

class GenericBeforeBaseModelWarning(Warning):
    pass

class PydanticExperimentalWarning(Warning):
    """A Pydantic specific experimental functionality warning.

    This warning is raised when using experimental functionality in Pydantic.
    It is raised to warn users that the functionality may change or be removed in future versions of Pydantic.
    """
