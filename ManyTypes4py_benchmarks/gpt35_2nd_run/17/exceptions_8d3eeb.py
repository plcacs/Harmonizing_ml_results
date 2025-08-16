from enum import Enum, auto
from typing import Any, Dict, List, Optional

class ErrorCode(Enum):
    BAD_REQUEST = auto()
    REQUEST_VARIABLE_MISSING = auto()
    REQUEST_VARIABLE_INVALID = auto()
    INVALID_JSON = auto()
    BAD_IMAGE = auto()
    REALM_UPLOAD_QUOTA = auto()
    BAD_NARROW = auto()
    CANNOT_DEACTIVATE_LAST_USER = auto()
    MISSING_HTTP_EVENT_HEADER = auto()
    STREAM_DOES_NOT_EXIST = auto()
    UNAUTHORIZED_PRINCIPAL = auto()
    UNSUPPORTED_WEBHOOK_EVENT_TYPE = auto()
    ANOMALOUS_WEBHOOK_PAYLOAD = auto()
    BAD_EVENT_QUEUE_ID = auto()
    CSRF_FAILED = auto()
    INVITATION_FAILED = auto()
    INVALID_ZULIP_SERVER = auto()
    INVALID_PUSH_DEVICE_TOKEN = auto()
    INVALID_REMOTE_PUSH_DEVICE_TOKEN = auto()
    INVALID_MARKDOWN_INCLUDE_STATEMENT = auto()
    REQUEST_CONFUSING_VAR = auto()
    INVALID_API_KEY = auto()
    INVALID_ZOOM_TOKEN = auto()
    UNKNOWN_ZOOM_USER = auto()
    UNAUTHENTICATED_USER = auto()
    NONEXISTENT_SUBDOMAIN = auto()
    RATE_LIMIT_HIT = auto()
    USER_DEACTIVATED = auto()
    REALM_DEACTIVATED = auto()
    REMOTE_SERVER_DEACTIVATED = auto()
    PASSWORD_AUTH_DISABLED = auto()
    PASSWORD_RESET_REQUIRED = auto()
    AUTHENTICATION_FAILED = auto()
    UNAUTHORIZED = auto()
    REQUEST_TIMEOUT = auto()
    MOVE_MESSAGES_TIME_LIMIT_EXCEEDED = auto()
    REACTION_ALREADY_EXISTS = auto()
    REACTION_DOES_NOT_EXIST = auto()
    SERVER_NOT_READY = auto()
    MISSING_REMOTE_REALM = auto()
    TOPIC_WILDCARD_MENTION_NOT_ALLOWED = auto()
    STREAM_WILDCARD_MENTION_NOT_ALLOWED = auto()
    REMOTE_BILLING_UNAUTHENTICATED_USER = auto()
    REMOTE_REALM_SERVER_MISMATCH_ERROR = auto()
    PUSH_NOTIFICATIONS_DISALLOWED = auto()
    EXPECTATION_MISMATCH = auto()
    SYSTEM_GROUP_REQUIRED = auto()
    CANNOT_DEACTIVATE_GROUP_IN_USE = auto()
    CANNOT_ADMINISTER_CHANNEL = auto()
    REMOTE_SERVER_VERIFICATION_SECRET_NOT_PREPARED = auto()
    HOSTNAME_ALREADY_IN_USE_BOUNCER_ERROR = auto()

class JsonableError(Exception):
    code: ErrorCode = ErrorCode.BAD_REQUEST
    data_fields: List[str] = []
    http_status_code: int = 400

    def __init__(self, msg: str) -> None:
        self._msg = msg

    @staticmethod
    def msg_format() -> str:
        return '{_msg}'

    @property
    def extra_headers(self) -> Dict[str, Any]:
        return {}

    @property
    def msg(self) -> str:
        format_data = dict(((f, getattr(self, f)) for f in self.data_fields), _msg=getattr(self, '_msg', None))
        return self.msg_format().format(**format_data)

    @property
    def data(self) -> Dict[str, Any]:
        return dict(((f, getattr(self, f)) for f in self.data_fields), code=self.code.name)

class UnauthorizedError(JsonableError):
    code: ErrorCode = ErrorCode.UNAUTHORIZED
    http_status_code: int = 401

    def __init__(self, msg: Optional[str] = None, www_authenticate: Optional[str] = None) -> None:
        if msg is None:
            msg = _('Not logged in: API authentication or user session required')
        super().__init__(msg)
        if www_authenticate is None:
            self.www_authenticate = 'Basic realm="zulip"'
        elif www_authenticate == 'session':
            self.www_authenticate = 'Session realm="zulip"'
        else:
            raise AssertionError('Invalid www_authenticate value!')

    @property
    def extra_headers(self) -> Dict[str, Any]:
        extra_headers_dict = super().extra_headers
        extra_headers_dict['WWW-Authenticate'] = self.www_authenticate
        return extra_headers_dict

class StreamDoesNotExistError(JsonableError):
    code: ErrorCode = ErrorCode.STREAM_DOES_NOT_EXIST
    data_fields: List[str] = ['stream']

    def __init__(self, stream: str) -> None:
        self.stream = stream

    @staticmethod
    def msg_format() -> str:
        return _("Channel '{stream}' does not exist")

# Add type annotations for the remaining classes as needed
