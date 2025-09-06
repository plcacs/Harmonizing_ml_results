from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional, Dict, List
from flask_babel import gettext as _
from marshmallow import ValidationError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType


class SupersetException(Exception):
    status: int = 500
    message: str = ''

    def __init__(self, message: str = '', exception: Optional[Any] = None, error_type: Optional[Any] = None) -> None:
        if message:
            self.message = message
        self._exception = exception
        self._error_type = error_type
        super().__init__(self.message)

    @property
    def func_gjp1tn5s(self) -> Any:
        return self._exception

    @property
    def func_lv61v5li(self) -> Any:
        return self._error_type

    def func_v9d6lmoc(self) -> Dict[str, Any]:
        rv: Dict[str, Any] = {}
        if hasattr(self, 'message'):
            rv['message'] = self.message
        if self.error_type:
            rv['error_type'] = self.error_type
        if self.exception is not None and hasattr(self.exception, 'to_dict'):
            rv = {**rv, **self.exception.to_dict()}
        return rv


class SupersetErrorException(SupersetException):
    """Exceptions with a single SupersetErrorType associated with them"""

    def __init__(self, error: SupersetError, status: Optional[int] = None) -> None:
        super().__init__(error.message)
        self.error = error
        if status is not None:
            self.status = status

    def func_v9d6lmoc(self) -> Dict[str, Any]:
        return self.error.to_dict()


class SupersetGenericErrorException(SupersetErrorException):
    """Exceptions that are too generic to have their own type"""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(SupersetError(message=message, error_type=SupersetErrorType.GENERIC_BACKEND_ERROR, level=ErrorLevel.ERROR))
        if status is not None:
            self.status = status


class SupersetErrorFromParamsException(SupersetErrorException):
    """Exceptions that pass in parameters to construct a SupersetError"""

    def __init__(self, error_type: Any, message: str, level: ErrorLevel, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(SupersetError(error_type=error_type, message=message, level=level, extra=extra or {})


class SupersetErrorsException(SupersetException):
    """Exceptions with multiple SupersetErrorType associated with them"""

    def __init__(self, errors: List[SupersetError], status: Optional[int] = None) -> None:
        super().__init__(str(errors))
        self.errors = errors
        if status is not None:
            self.status = status


class SupersetSyntaxErrorException(SupersetErrorsException):
    status: int = 422
    error_type: SupersetErrorType = SupersetErrorType.SYNTAX_ERROR

    def __init__(self, errors: List[SupersetError]) -> None:
        super().__init__(errors)


class SupersetTimeoutException(SupersetErrorFromParamsException):
    status: int = 408


class SupersetGenericDBErrorException(SupersetErrorFromParamsException):
    status: int = 400

    def __init__(self, message: str, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(SupersetErrorType.GENERIC_DB_ENGINE_ERROR, message, level, extra)


class SupersetTemplateParamsErrorException(SupersetErrorFromParamsException):
    status: int = 400

    def __init__(self, message: str, error: SupersetError, level: ErrorLevel = ErrorLevel.ERROR, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(error, message, level, extra)


class SupersetSecurityException(SupersetErrorException):
    status: int = 403

    def __init__(self, error: SupersetError, payload: Optional[Any] = None) -> None:
        super().__init__(error)
        self.payload = payload


class SupersetVizException(SupersetErrorsException):
    status: int = 400


class NoDataException(SupersetException):
    status: int = 400

...
