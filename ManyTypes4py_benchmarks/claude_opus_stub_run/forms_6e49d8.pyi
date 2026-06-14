import logging
from typing import Any

from django import forms
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.core.exceptions import ValidationError
from django.http import HttpRequest

from markupsafe import Markup
from two_factor.forms import AuthenticationTokenForm as TwoFactorAuthenticationTokenForm

from zerver.lib.rate_limiter import RateLimitedObject
from zerver.models import Realm, UserProfile

MIT_VALIDATION_ERROR: Markup
INVALID_ACCOUNT_CREDENTIALS_ERROR: str
DEACTIVATED_ACCOUNT_ERROR: str
PASSWORD_TOO_WEAK_ERROR: str

def email_is_not_mit_mailing_list(email: str) -> None: ...

class OverridableValidationError(ValidationError): ...

def check_subdomain_available(subdomain: str, allow_reserved_subdomain: bool = False) -> None: ...
def email_not_system_bot(email: str) -> None: ...
def email_is_not_disposable(email: str) -> None: ...

class RealmDetailsForm(forms.Form):
    realm_subdomain: forms.CharField
    realm_type: forms.TypedChoiceField
    realm_default_language: forms.ChoiceField
    realm_name: forms.CharField
    realm_creation: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_realm_subdomain(self) -> str: ...

class RegistrationForm(RealmDetailsForm):
    MAX_PASSWORD_LENGTH: int
    full_name: forms.CharField
    password: forms.CharField
    is_demo_organization: forms.BooleanField
    enable_marketing_emails: forms.BooleanField
    email_address_visibility: forms.TypedChoiceField
    realm: Realm | None

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
    realm: Realm | None
    from_multiuse_invite: bool
    invited_as: int | None

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_email(self) -> str: ...

class RealmCreationForm(RealmDetailsForm):
    email: forms.EmailField

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LoggingSetPasswordForm(SetPasswordForm):
    new_password1: forms.CharField
    new_password2: forms.CharField

    def clean_new_password1(self) -> str: ...
    def save(self, commit: bool = True) -> UserProfile: ...

class ZulipPasswordResetForm(PasswordResetForm):
    def save(
        self,
        domain_override: str | None = None,
        subject_template_name: str = ...,
        email_template_name: str = ...,
        use_https: bool = False,
        token_generator: PasswordResetTokenGenerator = ...,
        from_email: str | None = None,
        request: HttpRequest | None = None,
        html_email_template_name: str | None = None,
        extra_email_context: dict[str, Any] | None = None,
    ) -> None: ...

class RateLimitedPasswordResetByEmail(RateLimitedObject):
    email: str

    def __init__(self, email: str) -> None: ...
    def key(self) -> str: ...
    def rules(self) -> list[tuple[int, int]]: ...

def rate_limit_password_reset_form_by_email(email: str) -> None: ...

class CreateUserForm(forms.Form):
    full_name: forms.CharField
    email: forms.EmailField

class OurAuthenticationForm(AuthenticationForm):
    logger: logging.Logger

    def clean(self) -> dict[str, Any]: ...
    def add_prefix(self, field_name: str) -> str: ...

class AuthenticationTokenForm(TwoFactorAuthenticationTokenForm):
    otp_token: forms.IntegerField

class MultiEmailField(forms.Field):
    def to_python(self, emails: str | None) -> list[str]: ...
    def validate(self, emails: list[str]) -> None: ...

class FindMyTeamForm(forms.Form):
    emails: MultiEmailField

    def clean_emails(self) -> list[str]: ...

class RealmRedirectForm(forms.Form):
    subdomain: forms.CharField

    def clean_subdomain(self) -> str: ...