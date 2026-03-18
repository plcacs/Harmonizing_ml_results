```python
import sys
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
from django import forms
from django.core.exceptions import ValidationError
from django.http import HttpRequest
from django.utils.functional import Promise
from markupsafe import Markup
from two_factor.forms import AuthenticationTokenForm as TwoFactorAuthenticationTokenForm
from zerver.models import Realm, UserProfile

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

MIT_VALIDATION_ERROR: Markup = ...
INVALID_ACCOUNT_CREDENTIALS_ERROR: Promise = ...
DEACTIVATED_ACCOUNT_ERROR: Promise = ...
PASSWORD_TOO_WEAK_ERROR: Promise = ...

def email_is_not_mit_mailing_list(email: str) -> None: ...

class OverridableValidationError(ValidationError):
    pass

def check_subdomain_available(subdomain: str, allow_reserved_subdomain: bool = ...) -> None: ...

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
    
    def __init__(self, *args: Any, realm_creation: bool, realm: Optional[Realm] = ..., **kwargs: Any) -> None: ...
    
    def clean_full_name(self) -> str: ...
    
    def clean_password(self) -> str: ...

class ToSForm(forms.Form):
    terms: forms.BooleanField
    enable_marketing_emails: forms.BooleanField
    email_address_visibility: forms.TypedChoiceField
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class HomepageForm(forms.Form):
    email: forms.EmailField
    
    def __init__(self, *args: Any, realm: Optional[Realm] = ..., from_multiuse_invite: bool = ..., invited_as: Optional[int] = ..., **kwargs: Any) -> None: ...
    
    def clean_email(self) -> str: ...

class RealmCreationForm(RealmDetailsForm):
    email: forms.EmailField
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class LoggingSetPasswordForm(forms.SetPasswordForm):
    new_password1: forms.CharField
    new_password2: forms.CharField
    
    def clean_new_password1(self) -> str: ...
    
    def save(self, commit: bool = ...) -> UserProfile: ...

class ZulipPasswordResetForm(forms.PasswordResetForm):
    def save(self, domain_override: Optional[str] = ..., subject_template_name: str = ..., email_template_name: str = ..., use_https: bool = ..., token_generator: Any = ..., from_email: Optional[str] = ..., request: Optional[HttpRequest] = ..., html_email_template_name: Optional[str] = ..., extra_email_context: Optional[Dict[str, Any]] = ...) -> None: ...

class RateLimitedPasswordResetByEmail:
    def __init__(self, email: str) -> None: ...
    
    def key(self) -> str: ...
    
    def rules(self) -> Any: ...

def rate_limit_password_reset_form_by_email(email: str) -> None: ...

class CreateUserForm(forms.Form):
    full_name: forms.CharField
    email: forms.EmailField

class OurAuthenticationForm(forms.AuthenticationForm):
    logger: Any = ...
    
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
```