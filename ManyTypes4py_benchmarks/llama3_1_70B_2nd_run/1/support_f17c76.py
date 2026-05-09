from typing import Literal, Optional

class SupportRequestForm(forms.Form):
    MAX_SUBJECT_LENGTH: int = 50
    request_subject: forms.CharField = forms.CharField(max_length=MAX_SUBJECT_LENGTH)
    request_message: forms.CharField = forms.CharField(widget=forms.Textarea)

class DemoRequestForm(forms.Form):
    MAX_INPUT_LENGTH: int = 50
    SORTED_ORG_TYPE_NAMES: list[str] = sorted([org_type['name'] for org_type in Realm.ORG_TYPES.values() if not org_type['hidden']])
    full_name: forms.CharField = forms.CharField(max_length=MAX_INPUT_LENGTH)
    email: forms.EmailField = forms.EmailField()
    role: forms.CharField = forms.CharField(max_length=MAX_INPUT_LENGTH)
    organization_name: forms.CharField = forms.CharField(max_length=MAX_INPUT_LENGTH)
    organization_type: forms.CharField = forms.CharField()
    organization_website: forms.URLField = forms.URLField(required=True, assume_scheme='https')
    expected_user_count: forms.CharField = forms.CharField(max_length=MAX_INPUT_LENGTH)
    message: forms.CharField = forms.CharField(widget=forms.Textarea)

class SalesRequestForm(forms.Form):
    MAX_INPUT_LENGTH: int = 50
    organization_website: forms.URLField = forms.URLField(required=True, assume_scheme='https')
    expected_user_count: forms.CharField = forms.CharField(max_length=MAX_INPUT_LENGTH)
    message: forms.CharField = forms.CharField(widget=forms.Textarea)

@zulip_login_required
@typed_endpoint_without_parameters
def support_request(request: HttpRequest) -> HttpResponse:
    ...

@typed_endpoint_without_parameters
def demo_request(request: HttpRequest) -> HttpResponse:
    ...

@zulip_login_required
@typed_endpoint_without_parameters
def sales_support_request(request: HttpRequest) -> HttpResponse:
    ...

def get_plan_type_string(plan_type: int) -> str:
    ...

def get_confirmations(types: list[str], object_ids: list[int], hostname: Optional[str] = None) -> list[dict]:
    ...

@dataclass
class SupportSelectOption:
    name: str
    value: int

def get_remote_plan_tier_options() -> list[SupportSelectOption]:
    ...

def get_realm_plan_type_options() -> list[SupportSelectOption]:
    ...

def get_realm_plan_type_options_for_discount() -> list[SupportSelectOption]:
    ...

def get_default_max_invites_for_plan_type(realm: Realm) -> int:
    ...

def check_update_max_invites(realm: Realm, new_max: int, default_max: int) -> bool:
    ...

ModifyPlan = Literal['downgrade_at_billing_cycle_end', 'downgrade_now_without_additional_licenses', 'downgrade_now_void_open_invoices', 'upgrade_plan_tier']
RemoteServerStatus = Literal['active', 'deactivated']

def shared_support_context() -> dict:
    ...

@require_server_admin
@typed_endpoint
def support(request: HttpRequest, *, realm_id: Optional[int] = None, plan_type: Optional[int] = None, monthly_discounted_price: Optional[int] = None, annual_discounted_price: Optional[int] = None, minimum_licenses: Optional[int] = None, required_plan_tier: Optional[int] = None, new_subdomain: Optional[str] = None, status: Optional[Literal['active', 'deactivated']] = None, billing_modality: Optional[int] = None, sponsorship_pending: Optional[bool] = None, approve_sponsorship: bool = False, modify_plan: Optional[ModifyPlan] = None, scrub_realm: bool = False, delete_user_by_id: Optional[int] = None, query: Optional[str] = None, org_type: Optional[int] = None, max_invites: Optional[int] = None, plan_end_date: Optional[str] = None, fixed_price: Optional[int] = None, sent_invoice_id: Optional[str] = None, delete_fixed_price_next_plan: bool = False) -> HttpResponse:
    ...

def get_remote_servers_for_support(email_to_search: Optional[str], uuid_to_search: Optional[str], hostname_to_search: Optional[str]) -> list[RemoteZulipServer]:
    ...

@require_server_admin
@typed_endpoint
def remote_servers_support(request: HttpRequest, *, query: Optional[str] = None, remote_server_id: Optional[int] = None, remote_realm_id: Optional[int] = None, monthly_discounted_price: Optional[int] = None, annual_discounted_price: Optional[int] = None, minimum_licenses: Optional[int] = None, required_plan_tier: Optional[int] = None, fixed_price: Optional[int] = None, sent_invoice_id: Optional[str] = None, sponsorship_pending: Optional[bool] = None, approve_sponsorship: bool = False, billing_modality: Optional[int] = None, plan_end_date: Optional[str] = None, modify_plan: Optional[ModifyPlan] = None, delete_fixed_price_next_plan: bool = False, remote_server_status: Optional[RemoteServerStatus] = None, complimentary_access_plan: Optional[str] = None) -> HttpResponse:
    ...
