from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode
import orjson
from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from pydantic import Json
from corporate.lib.decorator import authenticated_remote_realm_management_endpoint, authenticated_remote_server_management_endpoint
from corporate.models import CustomerPlan, get_current_plan_by_customer, get_customer_by_realm
from zerver.context_processors import get_realm_from_request, latest_info_context
from zerver.decorator import add_google_analytics, zulip_login_required
from zerver.lib.github import InvalidPlatformError, get_latest_github_release_download_link_for_platform
from zerver.lib.realm_description import get_realm_text_description
from zerver.lib.realm_icon import get_realm_icon_url
from zerver.lib.subdomains import is_subdomain_root_or_alias
from zerver.lib.typed_endpoint import typed_endpoint
from zerver.models import Realm

if TYPE_CHECKING:
    from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession

@add_google_analytics
def apps_view(request: HttpRequest, platform: Optional[str] = None) -> TemplateResponse:
    ...

def app_download_link_redirect(request: HttpRequest, platform: str) -> TemplateResponse:
    ...

def is_customer_on_free_trial(customer_plan: CustomerPlan) -> bool:
    ...

@dataclass
class PlansPageContext:
    on_free_trial: bool = False
    sponsorship_pending: bool = False
    is_sponsored: bool = False
    is_cloud_realm: bool = False
    is_self_hosted_realm: bool = False
    is_new_customer: bool = False
    on_free_tier: bool = False
    customer_plan: Optional[CustomerPlan] = None
    has_scheduled_upgrade: bool = False
    scheduled_upgrade_plan: Optional[CustomerPlan] = None
    requested_sponsorship_plan: Optional[str] = None
    billing_base_url: str = ''
    tier_self_hosted_basic: str = CustomerPlan.TIER_SELF_HOSTED_BASIC
    tier_self_hosted_business: str = CustomerPlan.TIER_SELF_HOSTED_BUSINESS
    tier_cloud_standard: str = CustomerPlan.TIER_CLOUD_STANDARD
    tier_cloud_plus: str = CustomerPlan.TIER_CLOUD_PLUS

@add_google_analytics
def plans_view(request: HttpRequest) -> TemplateResponse:
    ...

@add_google_analytics
@authenticated_remote_realm_management_endpoint
def remote_realm_plans_page(request: HttpRequest, billing_session: 'RemoteRealmBillingSession') -> TemplateResponse:
    ...

@add_google_analytics
@authenticated_remote_server_management_endpoint
def remote_server_plans_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession') -> TemplateResponse:
    ...

@add_google_analytics
def team_view(request: HttpRequest) -> TemplateResponse:
    ...

@add_google_analytics
def landing_view(request: HttpRequest, template_name: str) -> TemplateResponse:
    ...

@add_google_analytics
def hello_view(request: HttpRequest) -> TemplateResponse:
    ...

@add_google_analytics
def communities_view(request: HttpRequest) -> TemplateResponse:
    ...

@zulip_login_required
def invoices_page(request: HttpRequest) -> HttpResponseRedirect:
    ...

@authenticated_remote_realm_management_endpoint
def remote_realm_invoices_page(request: HttpRequest, billing_session: 'RemoteRealmBillingSession') -> HttpResponseRedirect:
    ...

@authenticated_remote_server_management_endpoint
def remote_server_invoices_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession') -> HttpResponseRedirect:
    ...

@zulip_login_required
@typed_endpoint
def customer_portal(request: HttpRequest, *, return_to_billing_page: bool = False, manual_license_management: bool = False, tier: Optional[str] = None, setup_payment_by_invoice: bool = False) -> HttpResponseRedirect:
    ...

@typed_endpoint
@authenticated_remote_realm_management_endpoint
def remote_realm_customer_portal(request: HttpRequest, billing_session: 'RemoteRealmBillingSession', *, return_to_billing_page: bool = False, manual_license_management: bool = False, tier: Optional[str] = None, setup_payment_by_invoice: bool = False) -> HttpResponseRedirect:
    ...

@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_customer_portal(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, return_to_billing_page: bool = False, manual_license_management: bool = False, tier: Optional[str] = None, setup_payment_by_invoice: bool = False) -> HttpResponseRedirect:
    ...
