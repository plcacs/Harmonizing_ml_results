import base64
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import wraps
from io import BytesIO
from typing import TYPE_CHECKING, Concatenate, TypeVar, cast, overload, Any, Optional, Union
from urllib.parse import urlsplit

import django_otp
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME, authenticate
from django.contrib.auth import login as django_login
from django.contrib.auth.decorators import user_passes_test as django_user_passes_test
from django.contrib.auth.models import AbstractBaseUser, AnonymousUser
from django.contrib.auth.views import redirect_to_login
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect, QueryDict
from django.http.multipartparser import MultiPartParser
from django.shortcuts import resolve_url
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.utils.crypto import constant_time_compare
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.views.decorators.csrf import csrf_exempt
from django_otp import user_has_device
from two_factor.utils import default_device
from typing_extensions import ParamSpec

from zerver.context_processors import get_valid_realm_from_request
from zerver.lib.exceptions import (
    AccessDeniedError,
    AnomalousWebhookPayloadError,
    InvalidAPIKeyError,
    InvalidAPIKeyFormatError,
    InvalidJSONError,
    JsonableError,
    OrganizationAdministratorRequiredError,
    OrganizationMemberRequiredError,
    OrganizationOwnerRequiredError,
    RealmDeactivatedError,
    UnauthorizedError,
    UnsupportedWebhookEventTypeError,
    UserDeactivatedError,
    WebhookError,
)
from zerver.lib.queue import queue_json_publish_rollback_unsafe
from zerver.lib.rate_limiter import is_local_addr, rate_limit_request_by_ip, rate_limit_user
from zerver.lib.request import RequestNotes
from zerver.lib.response import json_method_not_allowed
from zerver.lib.subdomains import get_subdomain, user_matches_subdomain
from zerver.lib.timestamp import datetime_to_timestamp, timestamp_to_datetime
from zerver.lib.typed_endpoint import typed_endpoint
from zerver.lib.users import is_2fa_verified
from zerver.lib.utils import has_api_key_format
from zerver.lib.webhooks.common import notify_bot_owner_about_invalid_json
from zerver.models import UserProfile
from zerver.models.clients import get_client
from zerver.models.users import get_user_profile_by_api_key

if TYPE_CHECKING:
    from django.http.request import _ImmutableQueryDict

webhook_logger = logging.getLogger("zulip.zerver.webhooks")
webhook_unsupported_events_logger = logging.getLogger("zulip.zerver.webhooks.unsupported")
webhook_anomalous_payloads_logger = logging.getLogger("zulip.zerver.webhooks.anomalous")

ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")

def update_user_activity(
    request: HttpRequest, user_profile: UserProfile, query: Optional[str]
) -> None:
    # update_active_status also pushes to RabbitMQ, and it seems
    # redundant to log that here as well.
    if request.META["PATH_INFO"] == "/json/users/me/presence":
        return

    request_notes = RequestNotes.get_notes(request)
    if query is not None:
        pass
    elif request_notes.query is not None:
        query = request_notes.query
    else:
        query = request.META["PATH_INFO"]

    assert request_notes.client is not None
    event = {
        "query": query,
        "user_profile_id": user_profile.id,
        "time": datetime_to_timestamp(timezone_now()),
        "client_id": request_notes.client.id,
    }
    queue_json_publish_rollback_unsafe("user_activity", event, lambda event: None)

def require_post(
    func: Callable[Concatenate[HttpRequest, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, ParamT], HttpResponse]:
    @wraps(func)
    def wrapper(
        request: HttpRequest, /, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> HttpResponse:
        if request.method != "POST":
            err_method = request.method
            logging.warning(
                "Method Not Allowed (%s): %s",
                err_method,
                request.path,
                extra={"status_code": 405, "request": request},
            )
            if RequestNotes.get_notes(request).error_format == "JSON":
                return json_method_not_allowed(["POST"])
            else:
                return TemplateResponse(
                    request, "4xx.html", context={"status_code": 405}, status=405
                )
        return func(request, *args, **kwargs)

    return wrapper

def require_realm_owner(
    func: Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse]:
    @wraps(func)
    def wrapper(
        request: HttpRequest,
        user_profile: UserProfile,
        /,
        *args: ParamT.args,
        **kwargs: ParamT.kwargs,
    ) -> HttpResponse:
        if not user_profile.is_realm_owner:
            raise OrganizationOwnerRequiredError
        return func(request, user_profile, *args, **kwargs)

    return wrapper

def require_realm_admin(
    func: Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse]:
    @wraps(func)
    def wrapper(
        request: HttpRequest,
        user_profile: UserProfile,
        /,
        *args: ParamT.args,
        **kwargs: ParamT.kwargs,
    ) -> HttpResponse:
        if not user_profile.is_realm_admin:
            raise OrganizationAdministratorRequiredError
        return func(request, user_profile, *args, **kwargs)

    return wrapper

def check_if_user_can_manage_default_streams(
    func: Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse]:
    @wraps(func)
    def wrapper(
        request: HttpRequest,
        user_profile: UserProfile,
        /,
        *args: ParamT.args,
        **kwargs: ParamT.kwargs,
    ) -> HttpResponse:
        if not user_profile.can_manage_default_streams():
            raise OrganizationAdministratorRequiredError
        return func(request, user_profile, *args, **kwargs)

    return wrapper

def require_organization_member(
    func: Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse]:
    @wraps(func)
    def wrapper(
        request: HttpRequest,
        user_profile: UserProfile,
        /,
        *args: ParamT.args,
        **kwargs: ParamT.kwargs,
    ) -> HttpResponse:
        if user_profile.role > UserProfile.ROLE_MEMBER:
            raise OrganizationMemberRequiredError
        return func(request, user_profile, *args, **kwargs)

    return wrapper

def require_billing_access(
    func: Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, UserProfile, ParamT], HttpResponse]:
    @wraps(func)
    def wrapper(
        request: HttpRequest,
        user_profile: UserProfile,
        /,
        *args: ParamT.args,
        **kwargs: ParamT.kwargs,
    ) -> HttpResponse:
        if not user_profile.has_billing_access:
            raise JsonableError(_("Must be a billing administrator or an organization owner"))
        return func(request, user_profile, *args, **kwargs)

    return wrapper

def process_client(
    request: HttpRequest,
    user: Optional[Union[UserProfile, AnonymousUser]] = None,
    *,
    is_browser_view: bool = False,
    client_name: Optional[str] = None,
    query: Optional[str] = None,
) -> None:
    request_notes = RequestNotes.get_notes(request)
    if client_name is None:
        client_name = request_notes.client_name

    assert client_name is not None

    if is_browser_view and not client_name.startswith("Zulip"):
        client_name = "website"

    request_notes.client = get_client(client_name)
    if user is not None and user.is_authenticated:
        update_user_activity(request, user, query)

def validate_api_key(
    request: HttpRequest,
    role: Optional[str],
    api_key: str,
    allow_webhook_access: bool = False,
    client_name: Optional[str] = None,
) -> UserProfile:
    api_key = api_key.strip()
    if role is not None:
        role = role.strip()

    user_profile = access_user_by_api_key(request, api_key, email=role)
    if user_profile.is_incoming_webhook and not allow_webhook_access:
        raise JsonableError(_("This API is not available to incoming webhook bots."))

    request.user = user_profile
    process_client(request, user_profile, client_name=client_name)

    return user_profile

def validate_account_and_subdomain(request: HttpRequest, user_profile: UserProfile) -> None:
    if user_profile.realm.deactivated:
        raise RealmDeactivatedError
    if not user_profile.is_active:
        raise UserDeactivatedError

    remote_addr = request.META.get("REMOTE_ADDR", None)
    server_name = request.META.get("SERVER_NAME", None)

    if (
        settings.RUNNING_INSIDE_TORNADO
        and remote_addr == "127.0.0.1"
        and server_name == "127.0.0.1"
    ):
        return
    if remote_addr == "127.0.0.1" and server_name == "localhost":
        return

    if user_matches_subdomain(get_subdomain(request), user_profile):
        return

    logging.warning(
        "User %s (%s) attempted to access API on wrong subdomain (%s)",
        user_profile.delivery_email,
        user_profile.realm.subdomain,
        get_subdomain(request),
    )
    raise JsonableError(_("Account is not associated with this subdomain"))

def access_user_by_api_key(
    request: HttpRequest, api_key: str, email: Optional[str] = None
) -> UserProfile:
    if not has_api_key_format(api_key):
        raise InvalidAPIKeyFormatError

    try:
        user_profile = get_user_profile_by_api_key(api_key)
    except UserProfile.DoesNotExist:
        raise InvalidAPIKeyError
    if email is not None and email.lower() != user_profile.delivery_email.lower():
        raise InvalidAPIKeyError

    validate_account_and_subdomain(request, user_profile)

    return user_profile

def log_unsupported_webhook_event(request: HttpRequest, summary: str) -> None:
    extra = {"request": request}
    webhook_unsupported_events_logger.exception(summary, stack_info=True, extra=extra)

def log_exception_to_webhook_logger(request: HttpRequest, err: Exception) -> None:
    extra = {"request": request}
    if isinstance(err, AnomalousWebhookPayloadError):
        webhook_anomalous_payloads_logger.exception(err, extra=extra)
    elif isinstance(err, UnsupportedWebhookEventTypeError):
        webhook_unsupported_events_logger.exception(err, extra=extra)
    else:
        webhook_logger.exception(err, stack_info=True, extra=extra)

def full_webhook_client_name(raw_client_name: Optional[str] = None) -> Optional[str]:
    if raw_client_name is None:
        return None
    return f"Zulip{raw_client_name}Webhook"

def webhook_view(
    webhook_client_name: str,
    notify_bot_owner_on_invalid_json: bool = True,
    all_event_types: Optional[Sequence[str]] = None,
) -> Callable[[Callable[..., HttpResponse]], Callable[..., HttpResponse]]:
    def _wrapped_view_func(view_func: Callable[..., HttpResponse]) -> Callable[..., HttpResponse]:
        @csrf_exempt
        @wraps(view_func)
        @typed_endpoint
        def _wrapped_func_arguments(
            request: HttpRequest, /, *args: object, api_key: str, **kwargs: object
        ) -> HttpResponse:
            user_profile = validate_api_key(
                request,
                None,
                api_key,
                allow_webhook_access=True,
                client_name=full_webhook_client_name(webhook_client_name),
            )

            request_notes = RequestNotes.get_notes(request)
            request_notes.is_webhook_view = True

            rate_limit_user(request, user_profile, domain="api_by_user")
            try:
                return view_func(request, user_profile, *args, **kwargs)
            except Exception as err:
                if not isinstance(err, JsonableError):
                    log_exception_to_webhook_logger(request, err)
                elif isinstance(err, WebhookError):
                    err.webhook_name = webhook_client_name
                    log_exception_to_webhook_logger(request, err)
                elif isinstance(err, InvalidJSONError) and notify_bot_owner_on_invalid_json:
                    notify_bot_owner_about_invalid_json(user_profile, webhook_client_name)

                raise err

        _wrapped_func_arguments._all_event_types = all_event_types  # type: ignore[attr-defined]
        return _wrapped_func_arguments

    return _wrapped_view_func

def zulip_redirect_to_login(
    request: HttpRequest,
    login_url: Optional[str] = None,
    redirect_field_name: str = REDIRECT_FIELD_NAME,
) -> HttpResponseRedirect:
    path = request.build_absolute_uri()
    resolved_login_url = resolve_url(login_url or settings.LOGIN_URL)
    login_scheme, login_netloc = urlsplit(resolved_login_url)[:2]
    current_scheme, current_netloc = urlsplit(path)[:2]
    if (not login_scheme or login_scheme == current_scheme) and (
        not login_netloc or login_netloc == current_netloc
    ):
        path = request.get_full_path()

    if path == "/":
        return HttpResponseRedirect(resolved_login_url)
    return redirect_to_login(path, resolved_login_url, redirect_field_name)

def user_passes_test(
    test_func: Callable[[HttpRequest], bool],
    login_url: Optional[str] = None,
    redirect_field_name: str = REDIRECT_FIELD_NAME,
) -> Callable[
    [Callable[Concatenate[HttpRequest, ParamT], HttpResponse]],
    Callable[Concatenate[HttpRequest, ParamT], HttpResponse],
]:
    def decorator(
        view_func: Callable[Concatenate[HttpRequest, ParamT], HttpResponse],
    ) -> Callable[Concatenate[HttpRequest, ParamT], HttpResponse]:
        @wraps(view_func)
        def _wrapped_view(
            request: HttpRequest, /, *args: ParamT.args, **kwargs: ParamT.kwargs
        ) -> HttpResponse:
            if test_func(request):
                return view_func(request, *args, **kwargs)
            return zulip_redirect_to_login(request, login_url, redirect_field_name)

        return _wrapped_view

    return decorator

def logged_in_and_active(request: HttpRequest) -> bool:
    if not request.user.is_authenticated:
        return False
    if not request.user.is_active:
        return False
    if request.user.realm.deactivated:
        return False
    return user_matches_subdomain(get_subdomain(request), request.user)

def do_two_factor_login(request: HttpRequest, user_profile: UserProfile) -> None:
    device = default_device(user_profile)
    if device:
        django_otp.login(request, device)

def do_login(request: HttpRequest, user_profile: UserProfile) -> None:
    realm = get_valid_realm_from_request(request)
    validated_user_profile = authenticate(
        request=request, username=user_profile.delivery_email, realm=realm, use_dummy_backend=True
    )
    if validated_user_profile is None or validated_user_profile != user_profile:
        raise AssertionError("do_login called for a user_profile that shouldn't be able to log in")

    assert isinstance(validated_user_profile, UserProfile)

    django_login(request, validated_user_profile)
    RequestNotes.get_notes(
        request
    ).requester_for_logs = validated_user_profile.format_requester_for_logs()
    process_client(request, validated_user_profile, is_browser_view=True)
    if settings.TWO_FACTOR_AUTHENTICATION_ENABLED:
        do_two_factor_login(request, validated_user_profile)

def log_view_func(
    view_func: Callable[Concatenate[HttpRequest, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, ParamT], HttpResponse]:
    @wraps(view_func)
    def _wrapped_view_func(
        request: HttpRequest, /, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> HttpResponse:
        RequestNotes.get_notes(request).query = view_func.__name__
        return view_func(request, *args, **kwargs)

    return _wrapped_view_func

def add_logging_data(
    view_func: Callable[Concatenate[HttpRequest, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, ParamT], HttpResponse]:
    @wraps(view_func)
    def _wrapped_view_func(
        request: HttpRequest, /, *args: ParamT.args, **kwargs: ParamT.kwargs
    ) -> HttpResponse:
        process_client(request, request.user, is_browser_view=True, query=view_func.__name__)

        if request.user.is_authenticated:
            rate_limit_user(request, request.user, domain="api_by_user")
        else:
            rate_limit_request_by_ip(request, domain="api_by_ip")

        return view_func(request, *args, **kwargs)

    return _wrapped_view_func

def human_users_only(
    view_func: Callable[Concatenate[HttpRequest, ParamT], HttpResponse],
) -> Callable[Concatenate[HttpRequest, ParamT], HttpResponse]:
    @wraps(view_func)
    def _