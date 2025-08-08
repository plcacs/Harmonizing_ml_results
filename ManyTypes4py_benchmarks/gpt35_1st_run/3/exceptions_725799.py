from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Optional

class PrefectException(Exception):
    pass

class CrashedRun(PrefectException):
    pass

class FailedRun(PrefectException):
    pass

class CancelledRun(PrefectException):
    pass

class PausedRun(PrefectException):
    def __init__(self, *args, state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

class UnfinishedRun(PrefectException):
    pass

class MissingFlowError(PrefectException):
    pass

class UnspecifiedFlowError(PrefectException):
    pass

class MissingResult(PrefectException):
    pass

class ScriptError(PrefectException):
    def __init__(self, user_exc, path):
        pass

class ParameterTypeError(PrefectException):
    def __init__(self, msg):
        pass

    @classmethod
    def from_validation_error(cls, exc):
        pass

class ParameterBindError(TypeError, PrefectException):
    def __init__(self, msg):
        pass

    @classmethod
    def from_bind_failure(cls, fn, exc, call_args, call_kwargs):
        pass

class SignatureMismatchError(PrefectException, TypeError):
    def __init__(self, msg):
        pass

    @classmethod
    def from_bad_params(cls, expected_params, provided_params):
        pass

class ObjectNotFound(PrefectException):
    def __init__(self, http_exc, help_message=None, *args, **kwargs):
        pass

    def __str__(self):
        pass

class ObjectAlreadyExists(PrefectException):
    def __init__(self, http_exc, *args, **kwargs):
        pass

class UpstreamTaskError(PrefectException):
    pass

class MissingContextError(PrefectException, RuntimeError):
    pass

class MissingProfileError(PrefectException, ValueError):
    pass

class ReservedArgumentError(PrefectException, TypeError):
    pass

class InvalidNameError(PrefectException, ValueError):
    pass

class PrefectSignal(BaseException):
    pass

class Abort(PrefectSignal):
    pass

class Pause(PrefectSignal):
    def __init__(self, *args, state=None, **kwargs):
        pass

class ExternalSignal(BaseException):
    pass

class TerminationSignal(ExternalSignal):
    def __init__(self, signal):
        pass

class PrefectHTTPStatusError(HTTPStatusError):
    @classmethod
    def from_httpx_error(cls, httpx_error):
        pass

class MappingLengthMismatch(PrefectException):
    pass

class MappingMissingIterable(PrefectException):
    pass

class BlockMissingCapabilities(PrefectException):
    pass

class ProtectedBlockError(PrefectException):
    pass

class InvalidRepositoryURLError(PrefectException):
    pass

class InfrastructureError(PrefectException):
    pass

class InfrastructureNotFound(PrefectException):
    pass

class InfrastructureNotAvailable(PrefectException):
    pass

class NotPausedError(PrefectException):
    pass

class FlowPauseTimeout(PrefectException):
    pass

class FlowRunWaitTimeout(PrefectException):
    pass

class PrefectImportError(ImportError):
    def __init__(self, message):
        pass

class SerializationError(PrefectException):
    pass

class ConfigurationError(PrefectException):
    pass

class ProfileSettingsValidationError(PrefectException):
    def __init__(self, errors):
        pass

class HashError(PrefectException):
    pass
