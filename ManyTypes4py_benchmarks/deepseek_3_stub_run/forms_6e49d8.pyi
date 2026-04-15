import logging
from email.headerregistry import Address
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
import DNS
from django import forms
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.core.exceptions import ValidationError
from django.http import HttpRequest
from django.utils.functional import Promise
from markupsafe import Markup
from two_factor.forms import AuthenticationTokenForm as TwoFactorAuthenticationTokenForm
from typing_extensions import override
from zerver.lib.exceptions import JsonableError, RateLimitedError
from zerver.lib.rate_limiter import RateLimitedObject
from zerver.models import Realm, UserProfile
from zerver.models.realm_audit_logs import RealmAuditLog

MIT_VALIDATION_ERROR: Markup = ...
INVALID_ACCOUNT_CREDENTIALS_ERROR: Promise = ...
DEACTIVATED_ACCOUNT_ERROR: Promise = ...
PASSWORD_TOO_WEAK_ERROR: Promise = ...

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
    
    def __init__(self, *args: Any, realm_creation: bool, **kwargs: Any) -> None: ...
    def clean_realm_subdomain(self) -> str: ...

class RegistrationForm(RealmDetailsForm):
    MAX_PASSWORD_LENGTH: int = ...
    full_name: forms.CharField
    password: forms.CharField
    is_demo_organization: forms.BooleanField
    enable_marketing_emails: forms.BooleanField
    email_address_visibility: forms.TypedChoiceField
    
    def __init__(self, *args: Any, realm_creation: bool, realm: Optional[Realm] = None, **kwargs: Any) -> None: ...
    def clean_full_name(self) -> str: ...
    def clean_password(self) -> str: ...

class ToSForm(forms.Form):
    terms: forms.BooleanField
    enable_marketing_emails: forms.BooleanField
    email_address_visibility: forms.TypedChoiceField
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class HomepageForm(forms.Form):
    email: forms.EmailField
    
    def __init__(self, *args: Any, realm: Optional[Realm] = None, from_multiuse_invite: bool = False, invited_as: Optional[int] = None, **kwargs: Any) -> None: ...
    def clean_email(self) -> str: ...

class RealmCreationForm(RealmDetailsForm):
    email: forms.EmailField
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LoggingSetPasswordForm(SetPasswordForm):
    new_password1: forms.CharField
    new_password2: forms.CharField
    
    def clean_new_password1(self) -> str: ...
    @override
    def save(self, commit: bool = True) -> UserProfile: ...

class ZulipPasswordResetForm(PasswordResetForm):
    @override
    def save(self, domain_override: Optional[str] = None, subject_template_name: str = 'registration/password_reset_subject.txt', email_template_name: str = 'registration/password_reset_email.html', use_https: bool = False, token_generator: PasswordResetTokenGenerator = ..., from_email: Optional[str] = None, request: Optional[HttpRequest] = None, html_email_template_name: Optional[str] = None, extra_email_context: Optional[Dict[str, Any]] = None) -> None: ...

class RateLimitedPasswordResetByEmail(RateLimitedObject):
    def __init__(self, email: str) -> None: ...
    @override
    def key(self) -> str: ...
    @override
    def rules(self) -> List[Dict[str, Any]]: ...

def rate_limit_password_reset_form_by_email(email: str) -> None: ...

class CreateUserForm(forms.Form):
    full_name: forms.CharField
    email: forms.EmailField

class OurAuthenticationForm(AuthenticationForm):
    logger: logging.Logger
    
    @override
    def clean(self) -> Dict[str, Any]: ...
    @override
    def add_prefix(self, field_name: str) -> str: ...

class AuthenticationTokenForm(TwoFactorAuthenticationTokenForm):
    otp_token: forms.IntegerField

class MultiEmailField(forms.Field):
    @override
    def to_python(self, emails: str) -> List[str]: ...
    @override
    def validate(self, emails: List[str]) -> None: ...

class FindMyTeamForm(forms.Form):
    emails: MultiEmailField
    
    def clean_emails(self) -> List[str]: ...

class RealmRedirectForm(forms.Form):
    subdomain: forms.CharField
    
    def clean_subdomain(self) -> str: ...