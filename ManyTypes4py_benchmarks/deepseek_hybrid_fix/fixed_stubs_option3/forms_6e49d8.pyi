import logging
import re
from email.headerregistry import Address
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
import DNS
from django import forms
from django.conf import settings
from django.contrib.auth import authenticate, password_validation
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.tokens import PasswordResetTokenGenerator, default_token_generator
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.http import HttpRequest
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from markupsafe import Markup
from two_factor.forms import AuthenticationTokenForm as TwoFactorAuthenticationTokenForm
from two_factor.utils import totp_digits
from typing_extensions import override
from zerver.actions.user_settings import do_change_password
from zerver.actions.users import do_send_password_reset_email
from zerver.lib.email_validation import email_allowed_for_realm, email_reserved_for_system_bots_error, validate_is_not_disposable
from zerver.lib.exceptions import JsonableError, RateLimitedError
from zerver.lib.i18n import get_language_list
from zerver.lib.name_restrictions import is_reserved_subdomain
from zerver.lib.rate_limiter import RateLimitedObject, rate_limit_request_by_ip
from zerver.lib.subdomains import get_subdomain, is_root_domain_available
from zerver.lib.users import check_full_name
from zerver.models import Realm, UserProfile
from zerver.models.realm_audit_logs import RealmAuditLog
from zerver.models.realms import DisposableEmailError, DomainNotAllowedForRealmError, EmailContainsPlusError, get_realm
from zerver.models.users import get_user_by_delivery_email, is_cross_realm_bot_email
from zproject.backends import check_password_strength, email_auth_enabled, email_belongs_to_ldap
from django.utils.functional import Promise

MIT_VALIDATION_ERROR: Markup
INVALID_ACCOUNT_CREDENTIALS_ERROR: Promise
DEACTIVATED_ACCOUNT_ERROR: Promise
PASSWORD_TOO_WEAK_ERROR: Promise

def email_is_not_mit_mailing_list(email: str) -> None: ...

class OverridableValidationError(ValidationError): ...

def check_subdomain_available(subdomain: str, allow_reserved_subdomain: bool = ...) -> None: ...

def email_not_system_bot(email: str) -> None: ...

def email_is_not_disposable(email: str) -> None: ...

class RealmDetailsForm(forms.Form):
    realm_subdomain: forms.CharField
    realm_type: forms.TypedChoiceField
    realm_default_language: forms.ChoiceField
    realm_name: forms.CharField

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_realm_subdomain(self) -> str: ...

class RegistrationForm(RealmDetailsForm):
    MAX_PASSWORD_LENGTH: int
    full_name: forms.CharField
    password: forms.CharField
    is_demo_organization: forms.BooleanField
    enable_marketing_emails: forms.BooleanField
    email_address_visibility: forms.TypedChoiceField

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_full_name(self) -> str: ...
    def clean_password(self) -> str: ...

class ToSForm(forms.Form):
    terms: forms.BooleanField
    enable_marketing_emails: forms.BooleanField
    email_address_visibility: forms.TypedChoiceField

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class HomepageForm(forms.Form):
    email: forms.EmailField

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_email(self) -> str: ...

class RealmCreationForm(RealmDetailsForm):
    email: forms.EmailField

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LoggingSetPasswordForm(SetPasswordForm):
    new_password1: forms.CharField
    new_password2: forms.CharField

    def clean_new_password1(self) -> str: ...
    def save(self, commit: bool = ...) -> UserProfile: ...

class ZulipPasswordResetForm(PasswordResetForm):
    def save(self, domain_override: Optional[str] = ..., subject_template_name: str = ..., email_template_name: str = ..., use_https: bool = ..., token_generator: PasswordResetTokenGenerator = ..., from_email: Optional[str] = ..., request: Optional[HttpRequest] = ..., html_email_template_name: Optional[str] = ..., extra_email_context: Optional[Dict[str, Any]] = ...) -> None: ...

class RateLimitedPasswordResetByEmail(RateLimitedObject):
    email: str

    def __init__(self, email: str) -> None: ...
    def key(self) -> str: ...
    def rules(self) -> List[Dict[str, Any]]: ...

def rate_limit_password_reset_form_by_email(email: str) -> None: ...

class CreateUserForm(forms.Form):
    full_name: forms.CharField
    email: forms.EmailField

class OurAuthenticationForm(AuthenticationForm):
    logger: logging.Logger

    def clean(self) -> Dict[str, Any]: ...
    def add_prefix(self, field_name: str) -> str: ...

class AuthenticationTokenForm(TwoFactorAuthenticationTokenForm):
    otp_token: forms.IntegerField

class MultiEmailField(forms.Field):
    def to_python(self, emails: str) -> List[str]: ...
    def validate(self, emails: List[str]) -> None: ...

class FindMyTeamForm(forms.Form):
    emails: MultiEmailField

    def clean_emails(self) -> List[str]: ...

class RealmRedirectForm(forms.Form):
    subdomain: forms.CharField

    def clean_subdomain(self) -> str: ...