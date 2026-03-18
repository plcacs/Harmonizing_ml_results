from typing import Any, List, Optional
import logging
from django import forms
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.tokens import PasswordResetTokenGenerator, default_token_generator
from django.core.exceptions import ValidationError
from django.http import HttpRequest
from two_factor.forms import AuthenticationTokenForm as TwoFactorAuthenticationTokenForm
from zerver.lib.rate_limiter import RateLimitedObject
from zerver.models import Realm, UserProfile

MIT_VALIDATION_ERROR: Any = ...
INVALID_ACCOUNT_CREDENTIALS_ERROR: Any = ...
DEACTIVATED_ACCOUNT_ERROR: Any = ...
PASSWORD_TOO_WEAK_ERROR: Any = ...

def email_is_not_mit_mailing_list(email: str) -> None: ...
def check_subdomain_available(subdomain: str, allow_reserved_subdomain: bool = False) -> None: ...
def email_not_system_bot(email: str) -> None: ...
def email_is_not_disposable(email: str) -> None: ...

class OverridableValidationError(ValidationError): ...

class RealmDetailsForm(forms.Form):
    realm_subdomain: Any
    realm_type: Any
    realm_default_language: Any
    realm_name: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_realm_subdomain(self) -> str: ...

class RegistrationForm(RealmDetailsForm):
    MAX_PASSWORD_LENGTH: int
    full_name: Any
    password: Any
    is_demo_organization: Any
    enable_marketing_emails: Any
    email_address_visibility: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_full_name(self) -> str: ...
    def clean_password(self) -> str: ...

class ToSForm(forms.Form):
    terms: Any
    enable_marketing_emails: Any
    email_address_visibility: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class HomepageForm(forms.Form):
    email: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def clean_email(self) -> str: ...

class RealmCreationForm(RealmDetailsForm):
    email: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LoggingSetPasswordForm(SetPasswordForm):
    new_password1: Any
    new_password2: Any
    def clean_new_password1(self) -> str: ...
    def save(self, commit: bool = True) -> UserProfile: ...

class ZulipPasswordResetForm(PasswordResetForm):
    def save(
        self,
        domain_override: Any = ...,
        subject_template_name: str = 'registration/password_reset_subject.txt',
        email_template_name: str = 'registration/password_reset_email.html',
        use_https: bool = False,
        token_generator: PasswordResetTokenGenerator = default_token_generator,
        from_email: Any = ...,
        request: Optional[HttpRequest] = ...,
        html_email_template_name: Any = ...,
        extra_email_context: Any = ...,
    ) -> None: ...

class RateLimitedPasswordResetByEmail(RateLimitedObject):
    email: str
    def __init__(self, email: str) -> None: ...
    def key(self) -> str: ...
    def rules(self) -> Any: ...

def rate_limit_password_reset_form_by_email(email: str) -> None: ...

class CreateUserForm(forms.Form):
    full_name: Any
    email: Any

class OurAuthenticationForm(AuthenticationForm):
    logger: logging.Logger
    def clean(self) -> Any: ...
    def add_prefix(self, field_name: str) -> str: ...

class AuthenticationTokenForm(TwoFactorAuthenticationTokenForm):
    otp_token: Any

class MultiEmailField(forms.Field):
    def to_python(self, emails: Any) -> List[str]: ...
    def validate(self, emails: Any) -> None: ...

class FindMyTeamForm(forms.Form):
    emails: Any
    def clean_emails(self) -> List[str]: ...

class RealmRedirectForm(forms.Form):
    subdomain: Any
    def clean_subdomain(self) -> str: ...