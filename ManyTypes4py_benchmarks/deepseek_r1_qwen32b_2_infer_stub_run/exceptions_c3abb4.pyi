from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional, Dict, List, Union
from flask_babel import gettext as _
from marshmallow import ValidationError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType

class SupersetException(Exception):
    status: int
    message: str

    def __init__(self, message: str = '', exception: Optional[Exception] = None, error_type: Optional[SupersetErrorType] = None) -> None:
        ...

    @property
    def exception(self) -> Optional[Exception]:
        ...

    @property
    def error_type(self) -> Optional[SupersetErrorType]:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...

class SupersetErrorException(SupersetException):
    def __init__(self, error: SupersetError, status: Optional[int] = None) -> None:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...

class SupersetGenericErrorException(SupersetErrorException):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        ...

class SupersetErrorFromParamsException(SupersetErrorException):
    def __init__(self, error_type: SupersetErrorType, message: str, level: ErrorLevel, extra: Optional[Dict[str, Any]] = None) -> None:
        ...

class SupersetErrorsException(SupersetException):
    errors: List[SupersetError]

    def __init__(self, errors: List[SupersetError], status: Optional[int] = None) -> None:
        ...

class SupersetSyntaxErrorException(SupersetErrorsException):
    status: int
    error_type: SupersetErrorType

    def __init__(self, errors: List[SupersetError]) -> None:
        ...

class SupersetTimeoutException(SupersetErrorFromParamsException):
    status: int

    def __init__(self, error_type: SupersetErrorType, message: str, level: ErrorLevel, extra: Optional[Dict[str, Any]] = None) -> None:
        ...

class SupersetGenericDBErrorException(SupersetErrorFromParamsException):
    status: int

    def __init__(self, message: str, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict[str, Any]] = None) -> None:
        ...

class SupersetTemplateParamsErrorException(SupersetErrorFromParamsException):
    status: int

    def __init__(self, message: str, error: SupersetErrorType, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict[str, Any]] = None) -> None:
        ...

class SupersetSecurityException(SupersetErrorException):
    status: int
    payload: Optional[Any]

    def __init__(self, error: SupersetError, payload: Optional[Any] = None) -> None:
        ...

class SupersetVizException(SupersetErrorsException):
    status: int

    def __init__(self, errors: List[SupersetError], status: Optional[int] = None) -> None:
        ...

class NoDataException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class NullValueException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class SupersetTemplateException(SupersetException):
    def __init__(self, message: str = '') -> None:
        ...

class SpatialException(SupersetException):
    def __init__(self, message: str = '') -> None:
        ...

class CertificateException(SupersetException):
    message: str

    def __init__(self) -> None:
        ...

class DatabaseNotFound(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class MissingUserContextException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class QueryObjectValidationError(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class AdvancedDataTypeResponseError(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class InvalidPostProcessingError(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class CacheLoadError(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class QueryClauseValidationException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class DashboardImportException(SupersetException):
    def __init__(self, message: str = '') -> None:
        ...

class DatasetInvalidPermissionEvaluationException(SupersetException):
    def __init__(self, message: str = '') -> None:
        ...

class SerializationError(SupersetException):
    def __init__(self, message: str = '') -> None:
        ...

class InvalidPayloadFormatError(SupersetErrorException):
    status: int

    def __init__(self, message: str = _('Request payload has incorrect format')) -> None:
        ...

class InvalidPayloadSchemaError(SupersetErrorException):
    status: int

    def __init__(self, error: ValidationError) -> None:
        ...

class SupersetCancelQueryException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class QueryNotFoundException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class ColumnNotFoundException(SupersetException):
    status: int

    def __init__(self, message: str = '') -> None:
        ...

class SupersetMarshmallowValidationError(SupersetErrorException):
    status: int

    def __init__(self, exc: ValidationError, payload: Any) -> None:
        ...

class SupersetParseError(SupersetErrorException):
    status: int

    def __init__(self, sql: str, engine: Optional[Any] = None, message: Optional[str] = None, highlight: Optional[str] = None, line: Optional[int] = None, column: Optional[int] = None) -> None:
        ...

class OAuth2RedirectError(SupersetErrorException):
    status: int

    def __init__(self, url: str, tab_id: str, redirect_uri: str) -> None:
        ...

class OAuth2Error(SupersetErrorException):
    status: int

    def __init__(self, error: str) -> None:
        ...

class DisallowedSQLFunction(SupersetErrorException):
    status: int

    def __init__(self, functions: Union[str, List[str]]) -> None:
        ...

class CreateKeyValueDistributedLockFailedException(Exception):
    def __init__(self, message: str = '') -> None:
        ...

class DeleteKeyValueDistributedLockFailedException(Exception):
    def __init__(self, message: str = '') -> None:
        ...

class DatabaseNotFoundException(SupersetErrorException):
    status: int

    def __init__(self, message: str) -> None:
        ...

class TableNotFoundException(SupersetErrorException):
    status: int

    def __init__(self, message: str) -> None:
        ...