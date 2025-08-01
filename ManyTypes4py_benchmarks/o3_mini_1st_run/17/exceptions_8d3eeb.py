from enum import Enum, auto
from typing import Any, Dict, List, Optional
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from django_stubs_ext import StrPromise
from typing_extensions import override


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
        self._msg: str = msg

    @staticmethod
    def msg_format() -> str:
        return '{_msg}'

    @property
    def extra_headers(self) -> Dict[str, Any]:
        return {}

    @property
    def msg(self) -> str:
        format_data: Dict[str, Any] = {f: getattr(self, f) for f in self.data_fields}
        format_data['_msg'] = getattr(self, '_msg', None)
        return self.msg_format().format(**format_data)

    @property
    def data(self) -> Dict[str, Any]:
        return {**{f: getattr(self, f) for f in self.data_fields}, "code": self.code.name}

    @override
    def __str__(self) -> str:
        return self.msg


class UnauthorizedError(JsonableError):
    code: ErrorCode = ErrorCode.UNAUTHORIZED
    http_status_code: int = 401

    def __init__(self, msg: Optional[str] = None, www_authenticate: Optional[str] = None) -> None:
        if msg is None:
            msg = _('Not logged in: API authentication or user session required')
        super().__init__(msg)
        if www_authenticate is None:
            self.www_authenticate: str = 'Basic realm="zulip"'
        elif www_authenticate == 'session':
            self.www_authenticate = 'Session realm="zulip"'
        else:
            raise AssertionError('Invalid www_authenticate value!')

    @property
    @override
    def extra_headers(self) -> Dict[str, Any]:
        extra_headers_dict: Dict[str, Any] = super().extra_headers
        extra_headers_dict['WWW-Authenticate'] = self.www_authenticate
        return extra_headers_dict


class StreamDoesNotExistError(JsonableError):
    code: ErrorCode = ErrorCode.STREAM_DOES_NOT_EXIST
    data_fields: List[str] = ['stream']

    def __init__(self, stream: str) -> None:
        self.stream: str = stream

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Channel '{stream}' does not exist")


class StreamWithIDDoesNotExistError(JsonableError):
    code: ErrorCode = ErrorCode.STREAM_DOES_NOT_EXIST
    data_fields: List[str] = ['stream_id']

    def __init__(self, stream_id: int) -> None:
        self.stream_id: int = stream_id

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Channel with ID '{stream_id}' does not exist")


class IncompatibleParametersError(JsonableError):
    data_fields: List[str] = ['parameters']

    def __init__(self, parameters: List[str]) -> None:
        self.parameters: str = ', '.join(parameters)

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Unsupported parameter combination: {parameters}')


class CannotDeactivateLastUserError(JsonableError):
    code: ErrorCode = ErrorCode.CANNOT_DEACTIVATE_LAST_USER
    data_fields: List[str] = ['is_last_owner', 'entity']

    def __init__(self, is_last_owner: bool) -> None:
        self.is_last_owner: bool = is_last_owner
        self.entity: str = _('organization owner') if is_last_owner else _('user')

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Cannot deactivate the only {entity}.')


class InvalidMarkdownIncludeStatementError(JsonableError):
    code: ErrorCode = ErrorCode.INVALID_MARKDOWN_INCLUDE_STATEMENT
    data_fields: List[str] = ['include_statement']

    def __init__(self, include_statement: str) -> None:
        self.include_statement: str = include_statement

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid Markdown include statement: {include_statement}')


class RateLimitedError(JsonableError):
    code: ErrorCode = ErrorCode.RATE_LIMIT_HIT
    http_status_code: int = 429

    def __init__(self, secs_to_freedom: Optional[str] = None) -> None:
        self.secs_to_freedom: Optional[str] = secs_to_freedom

    @staticmethod
    @override
    def msg_format() -> str:
        return _('API usage exceeded rate limit')

    @property
    @override
    def extra_headers(self) -> Dict[str, Any]:
        extra_headers_dict: Dict[str, Any] = super().extra_headers
        if self.secs_to_freedom is not None:
            extra_headers_dict['Retry-After'] = self.secs_to_freedom
        return extra_headers_dict

    @property
    @override
    def data(self) -> Dict[str, Any]:
        data_dict: Dict[str, Any] = super().data
        data_dict['retry-after'] = self.secs_to_freedom
        return data_dict


class InvalidJSONError(JsonableError):
    code: ErrorCode = ErrorCode.INVALID_JSON

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Malformed JSON')


class OrganizationMemberRequiredError(JsonableError):
    code: ErrorCode = ErrorCode.UNAUTHORIZED_PRINCIPAL

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Must be an organization member')


class OrganizationAdministratorRequiredError(JsonableError):
    code: ErrorCode = ErrorCode.UNAUTHORIZED_PRINCIPAL

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Must be an organization administrator')


class OrganizationOwnerRequiredError(JsonableError):
    code: ErrorCode = ErrorCode.UNAUTHORIZED_PRINCIPAL

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Must be an organization owner')


class AuthenticationFailedError(JsonableError):
    code: ErrorCode = ErrorCode.AUTHENTICATION_FAILED
    http_status_code: int = 401

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Your username or password is incorrect')


class UserDeactivatedError(AuthenticationFailedError):
    code: ErrorCode = ErrorCode.USER_DEACTIVATED

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Account is deactivated')


class RealmDeactivatedError(AuthenticationFailedError):
    code: ErrorCode = ErrorCode.REALM_DEACTIVATED

    @staticmethod
    @override
    def msg_format() -> str:
        return _('This organization has been deactivated')


class RemoteServerDeactivatedError(AuthenticationFailedError):
    code: ErrorCode = ErrorCode.REALM_DEACTIVATED

    @staticmethod
    @override
    def msg_format() -> str:
        return _('The mobile push notification service registration for your server has been deactivated')


class PasswordAuthDisabledError(AuthenticationFailedError):
    code: ErrorCode = ErrorCode.PASSWORD_AUTH_DISABLED

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Password authentication is disabled in this organization')


class PasswordResetRequiredError(AuthenticationFailedError):
    code: ErrorCode = ErrorCode.PASSWORD_RESET_REQUIRED

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Your password has been disabled and needs to be reset')


class MarkdownRenderingError(Exception):
    pass


class InvalidAPIKeyError(JsonableError):
    code: ErrorCode = ErrorCode.INVALID_API_KEY
    http_status_code: int = 401

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid API key')


class InvalidAPIKeyFormatError(InvalidAPIKeyError):

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Malformed API key')


class WebhookError(JsonableError):
    data_fields: List[str] = ['webhook_name']

    def __init__(self) -> None:
        self.webhook_name: str = '(unknown)'


class UnsupportedWebhookEventTypeError(WebhookError):
    code: ErrorCode = ErrorCode.UNSUPPORTED_WEBHOOK_EVENT_TYPE
    http_status_code: int = 200
    data_fields: List[str] = ['webhook_name', 'event_type']

    def __init__(self, event_type: str) -> None:
        super().__init__()
        self.event_type: str = event_type

    @staticmethod
    @override
    def msg_format() -> str:
        return _("The '{event_type}' event isn't currently supported by the {webhook_name} webhook; ignoring")


class AnomalousWebhookPayloadError(WebhookError):
    code: ErrorCode = ErrorCode.ANOMALOUS_WEBHOOK_PAYLOAD

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Unable to parse request: Did {webhook_name} generate this event?')


class MissingAuthenticationError(JsonableError):
    code: ErrorCode = ErrorCode.UNAUTHENTICATED_USER
    http_status_code: int = 401

    def __init__(self) -> None:
        pass


class RemoteBillingAuthenticationError(JsonableError):
    code: ErrorCode = ErrorCode.REMOTE_BILLING_UNAUTHENTICATED_USER
    http_status_code: int = 401

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('User not authenticated')


class InvalidSubdomainError(JsonableError):
    code: ErrorCode = ErrorCode.NONEXISTENT_SUBDOMAIN
    http_status_code: int = 404

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid subdomain')


class ZephyrMessageAlreadySentError(Exception):

    def __init__(self, message_id: Any) -> None:
        self.message_id: Any = message_id


class InvitationError(JsonableError):
    code: ErrorCode = ErrorCode.INVITATION_FAILED
    data_fields: List[str] = ['errors', 'sent_invitations', 'license_limit_reached', 'daily_limit_reached']

    def __init__(self, msg: str, errors: Any, sent_invitations: Any, license_limit_reached: bool = False, daily_limit_reached: bool = False) -> None:
        self._msg: str = msg
        self.errors: Any = errors
        self.sent_invitations: Any = sent_invitations
        self.license_limit_reached: bool = license_limit_reached
        self.daily_limit_reached: bool = daily_limit_reached


class DirectMessageInitiationError(JsonableError):
    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('You do not have permission to initiate direct message conversations.')


class DirectMessagePermissionError(JsonableError):
    def __init__(self, is_nobody_group: bool) -> None:
        if is_nobody_group:
            msg = _('Direct messages are disabled in this organization.')
        else:
            msg = _('This conversation does not include any users who can authorize it.')
        super().__init__(msg)


class AccessDeniedError(JsonableError):
    http_status_code: int = 403

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Access denied')


class ResourceNotFoundError(JsonableError):
    http_status_code: int = 404


class ValidationFailureError(JsonableError):
    data_fields: List[str] = ['errors']

    def __init__(self, error: ValidationError) -> None:
        super().__init__(error.messages[0])
        self.errors: Any = error.message_dict


class MessageMoveError(JsonableError):
    code: ErrorCode = ErrorCode.MOVE_MESSAGES_TIME_LIMIT_EXCEEDED
    data_fields: List[str] = ['first_message_id_allowed_to_move', 'total_messages_in_topic', 'total_messages_allowed_to_move']

    def __init__(self, first_message_id_allowed_to_move: int, total_messages_in_topic: int, total_messages_allowed_to_move: int) -> None:
        self.first_message_id_allowed_to_move: int = first_message_id_allowed_to_move
        self.total_messages_in_topic: int = total_messages_in_topic
        self.total_messages_allowed_to_move: int = total_messages_allowed_to_move

    @staticmethod
    @override
    def msg_format() -> str:
        return _('You only have permission to move the {total_messages_allowed_to_move}/{total_messages_in_topic} most recent messages in this topic.')


class ReactionExistsError(JsonableError):
    code: ErrorCode = ErrorCode.REACTION_ALREADY_EXISTS

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Reaction already exists.')


class ReactionDoesNotExistError(JsonableError):
    code: ErrorCode = ErrorCode.REACTION_DOES_NOT_EXIST

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Reaction doesn't exist.")


class ApiParamValidationError(JsonableError):

    def __init__(self, msg: str, error_type: Any) -> None:
        super().__init__(msg)
        self.error_type: Any = error_type


class ServerNotReadyError(JsonableError):
    code: ErrorCode = ErrorCode.SERVER_NOT_READY
    http_status_code: int = 500


class RemoteRealmServerMismatchError(JsonableError):
    code: ErrorCode = ErrorCode.REMOTE_REALM_SERVER_MISMATCH_ERROR
    http_status_code: int = 403

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Your organization is registered to a different Zulip server. Please contact Zulip support for assistance in resolving this issue.')


class MissingRemoteRealmError(JsonableError):
    code: ErrorCode = ErrorCode.MISSING_REMOTE_REALM
    http_status_code: int = 403

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Organization not registered')


class StreamWildcardMentionNotAllowedError(JsonableError):
    code: ErrorCode = ErrorCode.STREAM_WILDCARD_MENTION_NOT_ALLOWED

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('You do not have permission to use channel wildcard mentions in this channel.')


class TopicWildcardMentionNotAllowedError(JsonableError):
    code: ErrorCode = ErrorCode.TOPIC_WILDCARD_MENTION_NOT_ALLOWED

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('You do not have permission to use topic wildcard mentions in this topic.')


class PreviousSettingValueMismatchedError(JsonableError):
    code: ErrorCode = ErrorCode.EXPECTATION_MISMATCH

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _("'old' value does not match the expected value.")


class SystemGroupRequiredError(JsonableError):
    code: ErrorCode = ErrorCode.SYSTEM_GROUP_REQUIRED
    data_fields: List[str] = ['setting_name']

    def __init__(self, setting_name: str) -> None:
        self.setting_name: str = setting_name

    @staticmethod
    @override
    def msg_format() -> str:
        return _("'{setting_name}' must be a system user group.")


class IncompatibleParameterValuesError(JsonableError):
    data_fields: List[str] = ['first_parameter', 'second_parameter']

    def __init__(self, first_parameter: str, second_parameter: str) -> None:
        self.first_parameter: str = first_parameter
        self.second_parameter: str = second_parameter

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Incompatible values for '{first_parameter}' and '{second_parameter}'.")


class CannotDeactivateGroupInUseError(JsonableError):
    code: ErrorCode = ErrorCode.CANNOT_DEACTIVATE_GROUP_IN_USE
    data_fields: List[str] = ['objections']

    def __init__(self, objections: Any) -> None:
        self.objections: Any = objections

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Cannot deactivate user group in use.')


class CannotAdministerChannelError(JsonableError):

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('You do not have permission to administer this channel.')


class CannotManageDefaultChannelError(JsonableError):

    def __init__(self) -> None:
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        return _('You do not have permission to change default channels.')