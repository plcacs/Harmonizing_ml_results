from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from typing import TYPE_CHECKING
from zerver.models import UserProfile
from zerver.lib.exceptions import JsonableError
from zerver.lib.response import json_success
from zerver.lib.typed_endpoint import typed_endpoint
from zerver.lib.typed_endpoint_validators import check_int_in
from corporate.models import CustomerPlan, get_current_plan_by_customer, get_customer_by_realm
from corporate.lib.decorator import authenticated_remote_realm_management_endpoint, authenticated_remote_server_management_endpoint
from zilencer.lib.remote_counts import MissingDataError
from zilencer.models import RemoteRealm, RemoteZulipServer

if TYPE_CHECKING:
    from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession

def billing_page(request: HttpRequest, *, success_message: str = '') -> HttpResponse:
    ...

def remote_realm_billing_page(request: HttpRequest, billing_session: 'RemoteRealmBillingSession', *, success_message: str = '') -> HttpResponse:
    ...

def remote_server_billing_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, success_message: str = '') -> HttpResponse:
    ...

def update_plan(request: HttpRequest, user: UserProfile, *, status: str = None, licenses: int = None, licenses_at_next_renewal: int = None, schedule: str = None, toggle_license_management: bool = False) -> HttpResponse:
    ...

def update_plan_for_remote_realm(request: HttpRequest, billing_session: 'RemoteRealmBillingSession', *, status: str = None, licenses: int = None, licenses_at_next_renewal: int = None, schedule: str = None, toggle_license_management: bool = False) -> HttpResponse:
    ...

def update_plan_for_remote_server(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, status: str = None, licenses: int = None, licenses_at_next_renewal: int = None, schedule: str = None, toggle_license_management: bool = False) -> HttpResponse:
    ...

def remote_server_deactivate_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, confirmed: bool = None) -> HttpResponse:
    ...
