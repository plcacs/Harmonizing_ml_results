from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from pydantic import Json
from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession
from zerver.models import UserProfile
from zilencer.lib.remote_counts import MissingDataError

def upgrade(request: HttpRequest, user: UserProfile, *, billing_modality: BillingModality, schedule: BillingSchedule, signed_seat_count: int, salt: str, license_management: LicenseManagement = None, licenses: Json = None, tier: str = CustomerPlan.TIER_CLOUD_STANDARD) -> HttpResponse:

def remote_realm_upgrade(request: HttpRequest, billing_session: RemoteRealmBillingSession, *, billing_modality: BillingModality, schedule: BillingSchedule, signed_seat_count: int, salt: str, license_management: LicenseManagement = None, licenses: Json = None, remote_server_plan_start_date: str = None, tier: str = CustomerPlan.TIER_SELF_HOSTED_BUSINESS) -> HttpResponse:

def remote_server_upgrade(request: HttpRequest, billing_session: RemoteServerBillingSession, *, billing_modality: BillingModality, schedule: BillingSchedule, signed_seat_count: int, salt: str, license_management: LicenseManagement = None, licenses: Json = None, remote_server_plan_start_date: str = None, tier: str = CustomerPlan.TIER_SELF_HOSTED_BUSINESS) -> HttpResponse:

def upgrade_page(request: HttpRequest, *, manual_license_management: bool = False, tier: str = CustomerPlan.TIER_CLOUD_STANDARD, setup_payment_by_invoice: bool = False) -> HttpResponse:

def remote_realm_upgrade_page(request: HttpRequest, billing_session: RemoteRealmBillingSession, *, manual_license_management: bool = False, success_message: str = '', tier: str = str(CustomerPlan.TIER_SELF_HOSTED_BUSINESS), setup_payment_by_invoice: bool = False) -> HttpResponse:

def remote_server_upgrade_page(request: HttpRequest, billing_session: RemoteServerBillingSession, *, manual_license_management: bool = False, success_message: str = '', tier: str = str(CustomerPlan.TIER_SELF_HOSTED_BUSINESS), setup_payment_by_invoice: bool = False) -> HttpResponse:
