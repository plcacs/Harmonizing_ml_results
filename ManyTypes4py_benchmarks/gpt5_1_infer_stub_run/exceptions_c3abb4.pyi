from typing import Any, Dict, List, Optional, Sequence, Union
from marshmallow import ValidationError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType

class SupersetException(Exception):
    status: int
    message: str
    def __init__(self, message: str = '', exception: Optional[Exception] = None, error_type: Optional[SupersetErrorType] = None) -> None: ...
    @property
    def exception(self) -> Optional[Exception]: ...
    @property
    def error_type(self) -> Optional[SupersetErrorType]: ...
    def to_dict(self) -> Dict[str, Any]: ...

class SupersetErrorException(SupersetException):
    error: SupersetError
    def __init__(self, error: SupersetError, status: Optional[int] = None) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...

class SupersetGenericErrorException(SupersetErrorException):
    def __init__(self, message: str, status: Optional[int] = None) -> None: ...

class SupersetErrorFromParamsException(SupersetErrorException):
    def __init__(self, error_type: SupersetErrorType, message: str, level: ErrorLevel, extra: Optional[Dict[str, Any]] = None) -> None: ...

class SupersetErrorsException(SupersetException):
    errors: List[SupersetError]
    def __init__(self, errors: List[SupersetError], status: Optional[int] = None) -> None: ...

class SupersetSyntaxErrorException(SupersetErrorsException):
    status: int
    error_type: SupersetErrorType
    def __init__(self, errors: List[SupersetError]) -> None: ...

class SupersetTimeoutException(SupersetErrorFromParamsException):
    status: int

class SupersetGenericDBErrorException(SupersetErrorFromParamsException):
    status: int
    def __init__(self, message: str, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict[str, Any]] = None) -> None: ...

class SupersetTemplateParamsErrorException(SupersetErrorFromParamsException):
    status: int
    def __init__(self, message: str, error: SupersetErrorType, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict[str, Any]] = None) -> None: ...

class SupersetSecurityException(SupersetErrorException):
    status: int
    payload: Optional[Dict[str, Any]]
    def __init__(self, error: SupersetError, payload: Optional[Dict[str, Any]] = None) -> None: ...

class SupersetVizException(SupersetErrorsException):
    status: int

class NoDataException(SupersetException):
    status: int

class NullValueException(SupersetException):
    status: int

class SupersetTemplateException(SupersetException): ...

class SpatialException(SupersetException): ...

class CertificateException(SupersetException):
    message: str

class DatabaseNotFound(SupersetException):
    status: int

class MissingUserContextException(SupersetException):
    status: int

class QueryObjectValidationError(SupersetException):
    status: int

class AdvancedDataTypeResponseError(SupersetException):
    status: int

class InvalidPostProcessingError(SupersetException):
    status: int

class CacheLoadError(SupersetException):
    status: int

class QueryClauseValidationException(SupersetException):
    status: int

class DashboardImportException(SupersetException): ...

class DatasetInvalidPermissionEvaluationException(SupersetException): ...

class SerializationError(SupersetException): ...

class InvalidPayloadFormatError(SupersetErrorException):
    status: int
    def __init__(self, message: str = 'Request payload has incorrect format') -> None: ...

class InvalidPayloadSchemaError(SupersetErrorException):
    status: int
    def __init__(self, error: ValidationError) -> None: ...

class SupersetCancelQueryException(SupersetException):
    status: int

class QueryNotFoundException(SupersetException):
    status: int

class ColumnNotFoundException(SupersetException):
    status: int

class SupersetMarshmallowValidationError(SupersetErrorException):
    status: int
    def __init__(self, exc: ValidationError, payload: Any) -> None: ...

class SupersetParseError(SupersetErrorException):
    status: int
    def __init__(self, sql: str, engine: Optional[str] = None, message: Optional[str] = None, highlight: Optional[str] = None, line: Optional[int] = None, column: Optional[int] = None) -> None: ...

class OAuth2RedirectError(SupersetErrorException):
    def __init__(self, url: str, tab_id: str, redirect_uri: str) -> None: ...

class OAuth2Error(SupersetErrorException):
    def __init__(self, error: Any) -> None: ...

class DisallowedSQLFunction(SupersetErrorException):
    def __init__(self, functions: Union[Sequence[str], str]) -> None: ...

class CreateKeyValueDistributedLockFailedException(Exception): ...

class DeleteKeyValueDistributedLockFailedException(Exception): ...

class DatabaseNotFoundException(SupersetErrorException):
    status: int
    def __init__(self, message: str) -> None: ...

class TableNotFoundException(SupersetErrorException):
    status: int
    def __init__(self, message: str) -> None: ...