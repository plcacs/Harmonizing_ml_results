from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional, Dict, List
from flask_babel import gettext as _
from marshmallow import ValidationError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType

class SupersetException(Exception):
    status: int
    message: str

    def __init__(self, message: str = '', exception: Optional[Exception] = None, error_type: Optional[Any] = None) -> None:
        ...

    @property
    def exception(self) -> Optional[Exception]:
        ...

    @property
    def error_type(self) -> Optional[Any]:
        ...

    def to_dict(self) -> Dict:
        ...

class SupersetErrorException(SupersetException):
    def __init__(self, error: SupersetError, status: Optional[int] = None) -> None:
        ...

    def to_dict(self) -> Dict:
        ...

class SupersetGenericErrorException(SupersetErrorException):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        ...

class SupersetErrorFromParamsException(SupersetErrorException):
    def __init__(self, error_type: SupersetErrorType, message: str, level: ErrorLevel, extra: Optional[Dict] = None) -> None:
        ...

class SupersetErrorsException(SupersetException):
    def __init__(self, errors: List[SupersetError], status: Optional[int] = None) -> None:
        ...

    def to_dict(self) -> Dict:
        ...

class SupersetSyntaxErrorException(SupersetErrorsException):
    status: int = 422
    error_type: SupersetErrorType = SupersetErrorType.SYNTAX_ERROR

    def __init__(self, errors: List[SupersetError]) -> None:
        ...

class SupersetTimeoutException(SupersetErrorFromParamsException):
    status: int = 408

class SupersetGenericDBErrorException(SupersetErrorFromParamsException):
    status: int = 400

    def __init__(self, message: str, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict] = None) -> None:
        ...

class SupersetTemplateParamsErrorException(SupersetErrorFromParamsException):
    status: int = 400

    def __init__(self, message: str, error: SupersetErrorType, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict] = None) -> None:
        ...

class SupersetSecurityException(SupersetErrorException):
    status: int = 403

    def __init__(self, error: SupersetError, payload: Optional[Any] = None) -> None:
        ...

class SupersetVizException(SupersetErrorsException):
    status: int = 400

class NoDataException(SupersetException):
    status: int = 400

class NullValueException(SupersetException):
    status: int = 400

class SupersetTemplateException(SupersetException):
    ...

class SpatialException(SupersetException):
    ...

class CertificateException(SupersetException):
    message: str = _('Invalid certificate')

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

class DashboardImportException(SupersetException):
    ...

class DatasetInvalidPermissionEvaluationException(SupersetException):
    ...

class SerializationError(SupersetException):
    ...

class InvalidPayloadFormatError(SupersetErrorException):
    status: int = 400

    def __init__(self, message: str = 'Request payload has incorrect format') -> None:
        ...

class InvalidPayloadSchemaError(SupersetErrorException):
    status: int = 422

    def __init__(self, error: ValidationError) -> None:
        ...

class SupersetCancelQueryException(SupersetException):
    status: int = 422

class QueryNotFoundException(SupersetException):
    status: int = 404

class ColumnNotFoundException(SupersetException):
    status: int = 404

class SupersetMarshmallowValidationError(SupersetErrorException):
    status: int = 422

    def __init__(self, exc: ValidationError, payload: Any) -> None:
        ...

class SupersetParseError(SupersetErrorException):
    status: int = 422

    def __init__(self, sql: str, engine: Optional[Any] = None, message: Optional[str] = None, highlight: Optional[str] = None, line: Optional[int] = None, column: Optional[int] = None) -> None:
        ...

class OAuth2RedirectError(SupersetErrorException):
    def __init__(self, url: str, tab_id: str, redirect_uri: str) -> None:
        ...

class OAuth2Error(SupersetErrorException):
    def __init__(self, error: Dict) -> None:
        ...

class DisallowedSQLFunction(SupersetErrorException):
    def __init__(self, functions: List[str]) -> None:
        ...

class CreateKeyValueDistributedLockFailedException(Exception):
    ...

class DeleteKeyValueDistributedLockFailedException(Exception):
    ...

class DatabaseNotFoundException(SupersetErrorException):
    status: int = 404

    def __init__(self, message: str) -> None:
        ...

class TableNotFoundException(SupersetErrorException):
    status: int = 404

    def __init__(self, message: str) -> None:
        ...