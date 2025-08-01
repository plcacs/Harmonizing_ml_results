#!/usr/bin/env python3
"""
Prefect-specific exceptions.
"""
import inspect
import traceback
from collections.abc import Iterable
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, Callable, Optional, List, Dict, Type, Union
from httpx._exceptions import HTTPStatusError
from pydantic import ValidationError
from typing_extensions import Self

if TYPE_CHECKING:
    from prefect.states import State

def _trim_traceback(tb: Optional[TracebackType],
                    remove_modules: Iterable[ModuleType]) -> Optional[TracebackType]:
    """
    Utility to remove frames from specific modules from a traceback.

    Only frames from the front of the traceback are removed. Once a traceback frame
    is reached that does not originate from `remove_modules`, it is returned.

    Args:
        tb: The traceback to trim.
        remove_modules: An iterable of module objects to remove.

    Returns:
        A traceback, or `None` if all traceback frames originate from an excluded module
    """
    strip_paths: List[str] = [
        module.__file__ for module in remove_modules if module.__file__ is not None
    ]
    while tb and any((module_path in str(tb.tb_frame.f_globals.get('__file__', ''))
                      for module_path in strip_paths)):
        tb = tb.tb_next
    return tb

def exception_traceback(exc: BaseException) -> str:
    """
    Convert an exception to a printable string with a traceback.
    """
    tb: traceback.TracebackException = traceback.TracebackException.from_exception(exc)
    return ''.join(list(tb.format()))

class PrefectException(Exception):
    """
    Base exception type for Prefect errors.
    """
    pass

class CrashedRun(PrefectException):
    """
    Raised when the result from a crashed run is retrieved.

    This occurs when a string is attached to the state instead of an exception or if
    the state's data is null.
    """
    pass

class FailedRun(PrefectException):
    """
    Raised when the result from a failed run is retrieved and an exception is not
    attached.

    This occurs when a string is attached to the state instead of an exception or if
    the state's data is null.
    """
    pass

class CancelledRun(PrefectException):
    """
    Raised when the result from a cancelled run is retrieved and an exception
    is not attached.

    This occurs when a string is attached to the state instead of an exception or if
    the state's data is null.
    """
    pass

class PausedRun(PrefectException):
    """
    Raised when the result from a paused run is retrieved.
    """
    def __init__(self, *args: Any, state: Optional["State"] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state: Optional["State"] = state

class UnfinishedRun(PrefectException):
    """
    Raised when the result from a run that is not finished is retrieved.

    For example, if a run is in a SCHEDULED, PENDING, CANCELLING, or RUNNING state.
    """
    pass

class MissingFlowError(PrefectException):
    """
    Raised when a given flow name is not found in the expected script.
    """
    pass

class UnspecifiedFlowError(PrefectException):
    """
    Raised when multiple flows are found in the expected script and no name is given.
    """
    pass

class MissingResult(PrefectException):
    """
    Raised when a result is missing from a state; often when result persistence is
    disabled and the state is retrieved from the API.
    """
    pass

class ScriptError(PrefectException):
    """
    Raised when a script errors during evaluation while attempting to load data.
    """
    def __init__(self, user_exc: BaseException, path: Any) -> None:
        import prefect.utilities.importtools
        message: str = f'Script at {str(path)!r} encountered an exception: {user_exc!r}'
        super().__init__(message)
        self.user_exc: BaseException = user_exc
        self.user_exc.__traceback__ = _trim_traceback(
            self.user_exc.__traceback__,
            remove_modules=[prefect.utilities.importtools]
        )

class ParameterTypeError(PrefectException):
    """
    Raised when a parameter does not pass Pydantic type validation.
    """
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

    @classmethod
    def from_validation_error(cls, exc: ValidationError) -> Self:
        bad_params: List[str] = [
            f"{'.'.join((str(item) for item in err['loc']))}: {err['msg']}"
            for err in exc.errors()
        ]
        msg: str = 'Flow run received invalid parameters:\n - ' + '\n - '.join(bad_params)
        return cls(msg)

class ParameterBindError(TypeError, PrefectException):
    """
    Raised when args and kwargs cannot be converted to parameters.
    """
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

    @classmethod
    def from_bind_failure(cls,
                          fn: Callable[..., Any],
                          exc: Exception,
                          call_args: List[Any],
                          call_kwargs: Dict[str, Any]) -> Self:
        fn_signature: str = str(inspect.signature(fn)).strip('()')
        base: str = f"Error binding parameters for function '{fn.__name__}': {exc}"
        signature: str = f"Function '{fn.__name__}' has signature '{fn_signature}'"
        received: str = f"received args: {call_args} and kwargs: {list(call_kwargs.keys())}"
        msg: str = f'{base}.\n{signature} but {received}.'
        return cls(msg)

class SignatureMismatchError(PrefectException, TypeError):
    """Raised when parameters passed to a function do not match its signature."""
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

    @classmethod
    def from_bad_params(cls, expected_params: Any, provided_params: Any) -> Self:
        msg: str = f'Function expects parameters {expected_params} but was provided with parameters {provided_params}'
        return cls(msg)

class ObjectNotFound(PrefectException):
    """
    Raised when the client receives a 404 (not found) from the API.
    """
    def __init__(self,
                 http_exc: HTTPStatusError,
                 help_message: Optional[str] = None,
                 *args: Any,
                 **kwargs: Any) -> None:
        self.http_exc: HTTPStatusError = http_exc
        self.help_message: Optional[str] = help_message
        super().__init__(help_message, *args, **kwargs)

    def __str__(self) -> str:
        return self.help_message or super().__str__()

class ObjectAlreadyExists(PrefectException):
    """
    Raised when the client receives a 409 (conflict) from the API.
    """
    def __init__(self, http_exc: HTTPStatusError, *args: Any, **kwargs: Any) -> None:
        self.http_exc: HTTPStatusError = http_exc
        super().__init__(*args, **kwargs)

class UpstreamTaskError(PrefectException):
    """
    Raised when a task relies on the result of another task but that task is not
    'COMPLETE'.
    """
    pass

class MissingContextError(PrefectException, RuntimeError):
    """
    Raised when a method is called that requires a task or flow run context to be
    active but one cannot be found.
    """
    pass

class MissingProfileError(PrefectException, ValueError):
    """
    Raised when a profile name does not exist.
    """
    pass

class ReservedArgumentError(PrefectException, TypeError):
    """
    Raised when a function used with Prefect has an argument with a name that is
    reserved for a Prefect feature.
    """
    pass

class InvalidNameError(PrefectException, ValueError):
    """
    Raised when a name contains characters that are not permitted.
    """
    pass

class PrefectSignal(BaseException):
    """
    Base type for signal-like exceptions that should never be caught by users.
    """
    pass

class Abort(PrefectSignal):
    """
    Raised when the API sends an 'ABORT' instruction during state proposal.

    Indicates that the run should exit immediately.
    """
    pass

class Pause(PrefectSignal):
    """
    Raised when a flow run is PAUSED and needs to exit for resubmission.
    """
    def __init__(self, *args: Any, state: Optional["State"] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state: Optional["State"] = state

class ExternalSignal(BaseException):
    """
    Base type for external signal-like exceptions that should never be caught by users.
    """
    pass

class TerminationSignal(ExternalSignal):
    """
    Raised when a flow run receives a termination signal.
    """
    def __init__(self, signal: Any) -> None:
        self.signal: Any = signal

class PrefectHTTPStatusError(HTTPStatusError):
    """
    Raised when client receives a `Response` that contains an HTTPStatusError.

    Used to include API error details in the error messages that the client provides users.
    """
    @classmethod
    def from_httpx_error(cls: Type[Self], httpx_error: HTTPStatusError) -> Self:
        """
        Generate a `PrefectHTTPStatusError` from an `httpx.HTTPStatusError`.
        """
        try:
            details: Any = httpx_error.response.json()
        except Exception:
            details = None
        error_message, *more_info = str(httpx_error).split('\n')
        if details:
            message_components: List[str] = [error_message, f'Response: {details}', *more_info]
        else:
            message_components = [error_message, *more_info]
        new_message: str = '\n'.join(message_components)
        return cls(new_message, request=httpx_error.request, response=httpx_error.response)

class MappingLengthMismatch(PrefectException):
    """
    Raised when attempting to call Task.map with arguments of different lengths.
    """
    pass

class MappingMissingIterable(PrefectException):
    """
    Raised when attempting to call Task.map with all static arguments.
    """
    pass

class BlockMissingCapabilities(PrefectException):
    """
    Raised when a block does not have required capabilities for a given operation.
    """
    pass

class ProtectedBlockError(PrefectException):
    """
    Raised when an operation is prevented due to block protection.
    """
    pass

class InvalidRepositoryURLError(PrefectException):
    """Raised when an incorrect URL is provided to a GitHub filesystem block."""
    pass

class InfrastructureError(PrefectException):
    """
    A base class for exceptions related to infrastructure blocks.
    """
    pass

class InfrastructureNotFound(PrefectException):
    """
    Raised when infrastructure is missing, likely because it has exited or been
    deleted.
    """
    pass

class InfrastructureNotAvailable(PrefectException):
    """
    Raised when infrastructure is not accessible from the current machine. For example,
    if a process was spawned on another machine it cannot be managed.
    """
    pass

class NotPausedError(PrefectException):
    """Raised when attempting to unpause a run that isn't paused."""
    pass

class FlowPauseTimeout(PrefectException):
    """Raised when a flow pause times out."""
    pass

class FlowRunWaitTimeout(PrefectException):
    """Raised when a flow run takes longer than a given timeout."""
    pass

class PrefectImportError(ImportError):
    """
    An error raised when a Prefect object cannot be imported due to a move or removal.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)

class SerializationError(PrefectException):
    """
    Raised when an object cannot be serialized.
    """
    pass

class ConfigurationError(PrefectException):
    """
    Raised when a configuration is invalid.
    """
    pass

class ProfileSettingsValidationError(PrefectException):
    """
    Raised when a profile settings are invalid.
    """
    def __init__(self, errors: Any) -> None:
        self.errors: Any = errors

class HashError(PrefectException):
    """Raised when hashing objects fails."""
    pass