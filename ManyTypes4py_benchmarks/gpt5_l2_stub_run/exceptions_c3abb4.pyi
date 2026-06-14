from __future__ import annotations

from typing import Any, Iterable, Optional
from marshmallow import ValidationError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType


class SupersetException(Exception):
    status: int = 500
    message: str = ""

    def __init__(
        self,
        message: str = "",
        exception: Optional[BaseException] = None,
        error_type: Optional[SupersetErrorType] = None,
    ) -> None: ...
    @property
    def exception(self) -> Optional[BaseException]: ...
    @property
    def error_type(self) -> Optional[SupersetErrorType]: ...
    def to_dict(self) -> dict[str, Any]: ...


class SupersetErrorException(SupersetException):
    """Exceptions with a single SupersetErrorType associated with them"""

    error: SupersetError

    def __init__(self, error: SupersetError, status: Optional[int] = None) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class SupersetGenericErrorException(SupersetErrorException):
    """Exceptions that are too generic to have their own type"""

    def __init__(self, message: str, status: Optional[int] = None) -> None: ...


class SupersetErrorFromParamsException(SupersetErrorException):
    """Exceptions that pass in parameters to construct a SupersetError"""

    def __init__(
        self,
        error_type: SupersetErrorType,
        message: str,
        level: ErrorLevel,
        extra: Optional[dict[str, Any]] = None,
    ) -> None: ...


class SupersetErrorsException(SupersetException):
    """Exceptions with multiple SupersetErrorType associated with them"""

    errors: list[SupersetError]

    def __init__(self, errors: list[SupersetError], status: Optional[int] = None) -> None: ...


class SupersetSyntaxErrorException(SupersetErrorsException):
    status: int = 422
    error_type: SupersetErrorType = SupersetErrorType.SYNTAX_ERROR

    def __init__(self, errors: list[SupersetError]) -> None: ...


class SupersetTimeoutException(SupersetErrorFromParamsException):
    status: int = 408


class SupersetGenericDBErrorException(SupersetErrorFromParamsException):
    status: int = 400

    def __init__(
        self,
        message: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        extra: Optional[dict[str, Any]] = None,
    ) -> None: ...


class SupersetTemplateParamsErrorException(SupersetErrorFromParamsException):
    status: int = 400

    def __init__(
        self,
        message: str,
        error: SupersetErrorType,
        level: ErrorLevel = ErrorLevel.ERROR,
        extra: Optional[dict[str, Any]] = None,
    ) -> None: ...


class SupersetSecurityException(SupersetErrorException):
    status: int = 403
    payload: Optional[dict[str, Any]]

    def __init__(self, error: SupersetError, payload: Optional[dict[str, Any]] = None) -> None: ...


class SupersetVizException(SupersetErrorsException):
    status: int = 400


class NoDataException(SupersetException):
    status: int = 400


class NullValueException(SupersetException):
    status: int = 400


class SupersetTemplateException(SupersetException): ...


class SpatialException(SupersetException): ...


class CertificateException(SupersetException):
    message: str


class DatabaseNotFound(SupersetException):
    status: int = 400


class MissingUserContextException(SupersetException):
    status: int = 422


class QueryObjectValidationError(SupersetException):
    status: int = 400


class AdvancedDataTypeResponseError(SupersetException):
    status: int = 400


class InvalidPostProcessingError(SupersetException):
    status: int = 400


class CacheLoadError(SupersetException):
    status: int = 404


class QueryClauseValidationException(SupersetException):
    status: int = 400


class DashboardImportException(SupersetException): ...


class DatasetInvalidPermissionEvaluationException(SupersetException): ...


class SerializationError(SupersetException): ...


class InvalidPayloadFormatError(SupersetErrorException):
    status: int = 400

    def __init__(self, message: str = "Request payload has incorrect format") -> None: ...


class InvalidPayloadSchemaError(SupersetErrorException):
    status: int = 422

    def __init__(self, error: ValidationError) -> None: ...


class SupersetCancelQueryException(SupersetException):
    status: int = 422


class QueryNotFoundException(SupersetException):
    status: int = 404


class ColumnNotFoundException(SupersetException):
    status: int = 404


class SupersetMarshmallowValidationError(SupersetErrorException):
    """
    Exception to be raised for Marshmallow validation errors.
    """

    status: int = 422

    def __init__(self, exc: ValidationError, payload: dict[str, Any]) -> None: ...


class SupersetParseError(SupersetErrorException):
    """
    Exception to be raised when we fail to parse SQL.
    """

    status: int = 422

    def __init__(
        self,
        sql: str,
        engine: Optional[str] = None,
        message: Optional[str] = None,
        highlight: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None: ...


class OAuth2RedirectError(SupersetErrorException):
    """
    Exception used to start OAuth2 dance for personal tokens.
    """

    def __init__(self, url: str, tab_id: str, redirect_uri: str) -> None: ...


class OAuth2Error(SupersetErrorException):
    """
    Exception for when OAuth2 goes wrong.
    """

    def __init__(self, error: Any) -> None: ...


class DisallowedSQLFunction(SupersetErrorException):
    """
    Disallowed function found on SQL statement
    """

    def __init__(self, functions: Iterable[str]) -> None: ...


class CreateKeyValueDistributedLockFailedException(Exception): ...


class DeleteKeyValueDistributedLockFailedException(Exception): ...


class DatabaseNotFoundException(SupersetErrorException):
    status: int = 404

    def __init__(self, message: str) -> None: ...


class TableNotFoundException(SupersetErrorException):
    status: int = 404

    def __init__(self, message: str) -> None: ...