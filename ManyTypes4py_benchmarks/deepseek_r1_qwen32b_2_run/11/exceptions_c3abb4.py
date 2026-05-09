from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional, List, Union, Dict
from flask_babel import gettext as _
from marshmallow import ValidationError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType

class SupersetException(Exception):
    status = 500
    message = ''

    def __init__(self, message: str = '', exception: Optional[Any] = None, error_type: Optional[Any] = None):
        if message:
            self.message = message
        self._exception = exception
        self._error_type = error_type
        super().__init__(self.message)

    @property
    def exception(self) -> Any:
        return self._exception

    @property
    def error_type(self) -> Any:
        return self._error_type

    def to_dict(self) -> Dict:
        rv = {}
        if hasattr(self, 'message'):
            rv['message'] = self.message
        if self.error_type:
            rv['error_type'] = self.error_type
        if self.exception is not None and hasattr(self.exception, 'to_dict'):
            rv = {**rv, **self.exception.to_dict()}
        return rv

class SupersetErrorException(SupersetException):
    def __init__(self, error: SupersetError, status: Optional[int] = None):
        super().__init__(error.message)
        self.error = error
        if status is not None:
            self.status = status

    def to_dict(self) -> Dict:
        return self.error.to_dict()

class SupersetGenericErrorException(SupersetErrorException):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(SupersetError(message=message, error_type=SupersetErrorType.GENERIC_BACKEND_ERROR, level=ErrorLevel.ERROR))
        if status is not None:
            self.status = status

class SupersetErrorFromParamsException(SupersetErrorException):
    def __init__(self, error_type: SupersetErrorType, message: str, level: ErrorLevel, extra: Optional[Dict] = None):
        super().__init__(SupersetError(error_type=error_type, message=message, level=level, extra=extra or {}))

class SupersetErrorsException(SupersetException):
    def __init__(self, errors: List[SupersetError], status: Optional[int] = None):
        super().__init__(str(errors))
        self.errors = errors
        if status is not None:
            self.status = status

class SupersetSyntaxErrorException(SupersetErrorsException):
    status = 422
    error_type = SupersetErrorType.SYNTAX_ERROR

    def __init__(self, errors: List[SupersetError]):
        super().__init__(errors)

class SupersetTimeoutException(SupersetErrorFromParamsException):
    status = 408

class SupersetGenericDBErrorException(SupersetErrorFromParamsException):
    status = 400

    def __init__(self, message: str, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict] = None):
        super().__init__(SupersetErrorType.GENERIC_DB_ENGINE_ERROR, message, level, extra)

class SupersetTemplateParamsErrorException(SupersetErrorFromParamsException):
    status = 400

    def __init__(self, message: str, error: SupersetErrorType, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict] = None):
        super().__init__(error, message, level, extra)

class SupersetSecurityException(SupersetErrorException):
    status = 403

    def __init__(self, error: SupersetError, payload: Optional[Any] = None):
        super().__init__(error)
        self.payload = payload

class SupersetVizException(SupersetErrorsException):
    status = 400

class NoDataException(SupersetException):
    status = 400

class NullValueException(SupersetException):
    status = 400

class SupersetTemplateException(SupersetException):
    pass

class SpatialException(SupersetException):
    pass

class CertificateException(SupersetException):
    message = _('Invalid certificate')

class DatabaseNotFound(SupersetException):
    status = 400

class MissingUserContextException(SupersetException):
    status = 422

class QueryObjectValidationError(SupersetException):
    status = 400

class AdvancedDataTypeResponseError(SupersetException):
    status = 400

class InvalidPostProcessingError(SupersetException):
    status = 400

class CacheLoadError(SupersetException):
    status = 404

class QueryClauseValidationException(SupersetException):
    status = 400

class DashboardImportException(SupersetException):
    pass

class DatasetInvalidPermissionEvaluationException(SupersetException):
    pass

class SerializationError(SupersetException):
    pass

class InvalidPayloadFormatError(SupersetErrorException):
    status = 400

    def __init__(self, message: str = 'Request payload has incorrect format'):
        error = SupersetError(message=message, error_type=SupersetErrorType.INVALID_PAYLOAD_FORMAT_ERROR, level=ErrorLevel.ERROR)
        super().__init__(error)

class InvalidPayloadSchemaError(SupersetErrorException):
    status = 422

    def __init__(self, error: ValidationError):
        for k, v in error.messages.items():
            if isinstance(v, defaultdict):
                error.messages[k] = dict(v)
        error = SupersetError(message='An error happened when validating the request', error_type=SupersetErrorType.INVALID_PAYLOAD_SCHEMA_ERROR, level=ErrorLevel.ERROR, extra={'messages': error.messages})
        super().__init__(error)

class SupersetCancelQueryException(SupersetException):
    status = 422

class QueryNotFoundException(SupersetException):
    status = 404

class ColumnNotFoundException(SupersetException):
    status = 404

class SupersetMarshmallowValidationError(SupersetErrorException):
    status = 422

    def __init__(self, exc: ValidationError, payload: Dict):
        error = SupersetError(message=_('The schema of the submitted payload is invalid.'), error_type=SupersetErrorType.MARSHMALLOW_ERROR, level=ErrorLevel.ERROR, extra={'messages': exc.messages, 'payload': payload})
        super().__init__(error)

class SupersetParseError(SupersetErrorException):
    status = 422

    def __init__(self, sql: str, engine: Optional[str] = None, message: Optional[str] = None, highlight: Optional[str] = None, line: Optional[int] = None, column: Optional[int] = None):
        if message is None:
            parts = [_('Error parsing')]
            if highlight:
                parts.append(_(" near '%(highlight)s'", highlight=highlight))
            if line:
                parts.append(_(' at line %(line)d', line=line))
                if column:
                    parts.append(f':{column}')
            message = ''.join(parts)
        error = SupersetError(message=message, error_type=SupersetErrorType.INVALID_SQL_ERROR, level=ErrorLevel.ERROR, extra={'sql': sql, 'engine': engine, 'line': line, 'column': column})
        super().__init__(error)

class OAuth2RedirectError(SupersetErrorException):
    def __init__(self, url: str, tab_id: str, redirect_uri: str):
        super().__init__(SupersetError(message="You don't have permission to access the data.", error_type=SupersetErrorType.OAUTH2_REDIRECT, level=ErrorLevel.WARNING, extra={'url': url, 'tab_id': tab_id, 'redirect_uri': redirect_uri}))

class OAuth2Error(SupersetErrorException):
    def __init__(self, error: Dict):
        super().__init__(SupersetError(message='Something went wrong while doing OAuth2', error_type=SupersetErrorType.OAUTH2_REDIRECT_ERROR, level=ErrorLevel.ERROR, extra={'error': error}))

class DisallowedSQLFunction(SupersetErrorException):
    def __init__(self, functions: Union[str, List[str]]):
        super().__init__(SupersetError(message=f'SQL statement contains disallowed function(s): {functions}', error_type=SupersetErrorType.SYNTAX_ERROR, level=ErrorLevel.ERROR))

class CreateKeyValueDistributedLockFailedException(Exception):
    pass

class DeleteKeyValueDistributedLockFailedException(Exception):
    pass

class DatabaseNotFoundException(SupersetErrorException):
    status = 404

    def __init__(self, message: str):
        super().__init__(SupersetError(message=message, error_type=SupersetErrorType.DATABASE_NOT_FOUND_ERROR, level=ErrorLevel.ERROR))

class TableNotFoundException(SupersetErrorException):
    status = 404

    def __init__(self, message: str):
        super().__init__(SupersetError(message=message, error_type=SupersetErrorType.TABLE_NOT_FOUND_ERROR, level=ErrorLevel.ERROR))