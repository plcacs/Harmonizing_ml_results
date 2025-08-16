def email_is_not_mit_mailing_list(email: str) -> None:
def check_subdomain_available(subdomain: str, allow_reserved_subdomain: bool = False) -> None:
def email_not_system_bot(email: str) -> None:
def email_is_not_disposable(email: str) -> None:
def RealmDetailsForm(forms.Form):
def RegistrationForm(RealmDetailsForm):
def ToSForm(forms.Form):
def HomepageForm(forms.Form):
def RealmCreationForm(RealmDetailsForm):
def LoggingSetPasswordForm(SetPasswordForm):
def ZulipPasswordResetForm(PasswordResetForm):
def rate_limit_password_reset_form_by_email(email: str) -> None:
def CreateUserForm(forms.Form):
def OurAuthenticationForm(AuthenticationForm):
def AuthenticationTokenForm(TwoFactorAuthenticationTokenForm):
def MultiEmailField(forms.Field):
def FindMyTeamForm(forms.Form):
def RealmRedirectForm(forms.Form):
