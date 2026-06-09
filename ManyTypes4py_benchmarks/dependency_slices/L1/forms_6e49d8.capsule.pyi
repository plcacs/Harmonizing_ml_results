from typing import Any

# === Unresolved dependency: DNS ===
# Used unresolved symbols: Base, Status, Type, dnslookup

# === Internal dependency: corporate.lib.registration ===
def check_spare_licenses_available_for_registering_new_user(realm, user_email_to_add, role): ...

# === Internal dependency: corporate.lib.stripe ===
class LicenseLimitError(Exception): ...

# === Third-party dependency: django ===
# Used symbols: forms

# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: django.contrib.auth ===
def authenticate(request = ..., **credentials) -> Any: ...

# === Third-party dependency: django.contrib.auth.forms ===
class AuthenticationForm(forms.Form):
    ...
class PasswordResetForm(forms.Form):
class SetPasswordForm(SetPasswordMixin, forms.Form):

# === Third-party dependency: django.contrib.auth.tokens ===
default_token_generator: PasswordResetTokenGenerator

# === Third-party dependency: django.core.exceptions ===
class ValidationError(Exception):
    def message_dict(self) -> Any: ...
    def messages(self) -> Any: ...

# === Third-party dependency: django.core.validators ===
validate_email: EmailValidator

# === Third-party dependency: django.utils.functional ===
def lazy(func, *resultclasses) -> Any: ...

# === Third-party dependency: django.utils.translation ===
def gettext(message) -> Any: ...
gettext_lazy: lazy

# === Third-party dependency: markupsafe ===
class Markup(str):
    ...

# === Unresolved dependency: two_factor.forms ===
# Used unresolved symbols: AuthenticationTokenForm

# === Unresolved dependency: two_factor.utils ===
# Used unresolved symbols: totp_digits

# === Internal dependency: zerver.actions.user_settings ===
def do_change_password(user_profile, password, commit=...): ...

# === Internal dependency: zerver.actions.users ===
def do_send_password_reset_email(email, realm, user_profile, *, token_generator=..., request=...): ...

# === Internal dependency: zerver.lib.email_validation ===
def validate_is_not_disposable(email): ...
def email_allowed_for_realm(email, realm): ...
def email_reserved_for_system_bots_error(email): ...

# === Internal dependency: zerver.lib.exceptions ===
class JsonableError(Exception): ...
class RateLimitedError(JsonableError): ...

# === Internal dependency: zerver.lib.i18n ===
def get_language_list(): ...

# === Internal dependency: zerver.lib.name_restrictions ===
def is_reserved_subdomain(subdomain): ...

# === Internal dependency: zerver.lib.rate_limiter ===
class RateLimitedObject(ABC):
    def key(self): ...
    def rules(self): ...
def rate_limit_request_by_ip(request, domain): ...

# === Internal dependency: zerver.lib.subdomains ===
def get_subdomain(request): ...
def is_root_domain_available(): ...

# === Internal dependency: zerver.lib.users ===
def check_full_name(full_name_raw, *, user_profile, realm): ...

# === Internal dependency: zerver.models ===
from zerver.models.realms import Realm as Realm
from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.realm_audit_logs ===
class RealmAuditLog(AbstractRealmAuditLog): ...

# === Internal dependency: zerver.models.realms ===
def get_realm(string_id): ...
class DomainNotAllowedForRealmError(Exception): ...
class DisposableEmailError(Exception): ...
class EmailContainsPlusError(Exception): ...

# === Internal dependency: zerver.models.users ===
def get_user_by_delivery_email(email, realm): ...
def is_cross_realm_bot_email(email): ...

# === Unresolved dependency: zproject.backends ===
# Used unresolved symbols: check_password_strength, email_auth_enabled, email_belongs_to_ldap