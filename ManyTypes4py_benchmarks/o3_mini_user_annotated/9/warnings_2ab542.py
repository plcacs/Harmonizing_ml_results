from __future__ import annotations

from typing import Any, Tuple, Optional
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
    message: str
    since: Tuple[int, int]
    expected_removal: Tuple[int, int]

    def __init__(
        self,
        message: str,
        *args: Any,
        since: Tuple[int, int],
        expected_removal: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(message, *args)
        self.message = message.rstrip('.')
        self.since = since
        self.expected_removal = expected_removal if expected_removal is not None else (since[0] + 1, 0)

    def __str__(self) -> str:
        message = (
            f'{self.message}. Deprecated in Pydantic V{self.since[0]}.{self.since[1]}'
            f' to be removed in V{self.expected_removal[0]}.{self.expected_removal[1]}.'
        )
        if self.since == (2, 0):
            message += (
                f' See Pydantic V2 Migration Guide at '
                f'https://errors.pydantic.dev/{version_short()}/migration/'
            )
        return message


class PydanticDeprecatedSince20(PydanticDeprecationWarning):
    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 0), expected_removal=(3, 0))


class PydanticDeprecatedSince26(PydanticDeprecationWarning):
    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 6), expected_removal=(3, 0))


class PydanticDeprecatedSince29(PydanticDeprecationWarning):
    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 9), expected_removal=(3, 0))


class PydanticDeprecatedSince210(PydanticDeprecationWarning):
    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 10), expected_removal=(3, 0))


class PydanticDeprecatedSince211(PydanticDeprecationWarning):
    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args, since=(2, 11), expected_removal=(3, 0))


class GenericBeforeBaseModelWarning(Warning):
    pass


class PydanticExperimentalWarning(Warning):
    pass
