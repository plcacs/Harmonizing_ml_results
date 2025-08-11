import logging
from typing import TYPE_CHECKING
from django.conf import settings
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from pydantic import Json
from corporate.lib.billing_types import BillingModality, BillingSchedule, LicenseManagement
from corporate.lib.decorator import authenticated_remote_realm_management_endpoint, authenticated_remote_server_management_endpoint
from corporate.models import CustomerPlan
from zerver.decorator import require_organization_member, zulip_login_required
from zerver.lib.response import json_success
from zerver.lib.typed_endpoint import typed_endpoint
from zerver.models import UserProfile
from zilencer.lib.remote_counts import MissingDataError
if TYPE_CHECKING:
    from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession
billing_logger = logging.getLogger('corporate.stripe')

@require_organization_member
@typed_endpoint
def upgrade(request: Union[str, django.http.HttpRequest, None], user: Union[str, zerver.models.UserProfile], *, billing_modality: Union[str, apistar.http.Request, bool], schedule: Union[str, apistar.http.Request, bool], signed_seat_count: Union[str, apistar.http.Request, bool], salt: Union[str, apistar.http.Request, bool], license_management: Union[None, str, apistar.http.Request, bool]=None, licenses: Union[None, str, apistar.http.Request, bool]=None, tier: Any=CustomerPlan.TIER_CLOUD_STANDARD) -> Union[str, apistar.http.Response, int, None]:
    from corporate.lib.stripe import BillingError, RealmBillingSession, UpgradeRequest
    try:
        upgrade_request = UpgradeRequest(billing_modality=billing_modality, schedule=schedule, signed_seat_count=signed_seat_count, salt=salt, license_management=license_management, licenses=licenses, tier=tier, remote_server_plan_start_date=None)
        billing_session = RealmBillingSession(user)
        data = billing_session.do_upgrade(upgrade_request)
        return json_success(request, data)
    except BillingError as e:
        billing_logger.warning('BillingError during upgrade: %s. user=%s, realm=%s (%s), billing_modality=%s, schedule=%s, license_management=%s, licenses=%s', e.error_description, user.id, user.realm.id, user.realm.string_id, billing_modality, schedule, license_management, licenses)
        raise e
    except Exception:
        billing_logger.exception('Uncaught exception in billing:', stack_info=True)
        error_message = BillingError.CONTACT_SUPPORT.format(email=settings.ZULIP_ADMINISTRATOR)
        error_description = 'uncaught exception during upgrade'
        raise BillingError(error_description, error_message)

@typed_endpoint
@authenticated_remote_realm_management_endpoint
def remote_realm_upgrade(request: Union[str, django.http.HttpRequest, None], billing_session: Union[str, grouper.models.base.session.Session, grouper.settings.Settings], *, billing_modality: Union[str, int, bytes], schedule: Union[str, int, bytes], signed_seat_count: Union[str, int, bytes], salt: Union[str, int, bytes], license_management: Union[None, str, int, bytes]=None, licenses: Union[None, str, int, bytes]=None, remote_server_plan_start_date: Union[None, str, int, bytes]=None, tier: Any=CustomerPlan.TIER_SELF_HOSTED_BUSINESS) -> Union[apistar.http.Response, django.http.response.HttpResponse]:
    from corporate.lib.stripe import BillingError, UpgradeRequest
    try:
        upgrade_request = UpgradeRequest(billing_modality=billing_modality, schedule=schedule, signed_seat_count=signed_seat_count, salt=salt, license_management=license_management, licenses=licenses, tier=tier, remote_server_plan_start_date=remote_server_plan_start_date)
        data = billing_session.do_upgrade(upgrade_request)
        return json_success(request, data)
    except BillingError as e:
        billing_logger.warning('BillingError during upgrade: %s. remote_realm=%s (%s), billing_modality=%s, schedule=%s, license_management=%s, licenses=%s', e.error_description, billing_session.remote_realm.id, billing_session.remote_realm.host, billing_modality, schedule, license_management, licenses)
        raise e
    except Exception:
        billing_logger.exception('Uncaught exception in billing:', stack_info=True)
        error_message = BillingError.CONTACT_SUPPORT.format(email=settings.ZULIP_ADMINISTRATOR)
        error_description = 'uncaught exception during upgrade'
        raise BillingError(error_description, error_message)

@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_upgrade(request: Union[str, django.http.HttpRequest, dict[str, typing.Any]], billing_session: Union[str, typing.Mapping, None, typing.Sequence], *, billing_modality: Union[str, int, django.http.HttpRequest], schedule: Union[str, int, django.http.HttpRequest], signed_seat_count: Union[str, int, django.http.HttpRequest], salt: Union[str, int, django.http.HttpRequest], license_management: Union[None, str, int, django.http.HttpRequest]=None, licenses: Union[None, str, int, django.http.HttpRequest]=None, remote_server_plan_start_date: Union[None, str, int, django.http.HttpRequest]=None, tier: Any=CustomerPlan.TIER_SELF_HOSTED_BUSINESS) -> Union[apistar.http.Response, django.http.response.HttpResponse, jumeaux.models.Request]:
    from corporate.lib.stripe import BillingError, UpgradeRequest
    try:
        upgrade_request = UpgradeRequest(billing_modality=billing_modality, schedule=schedule, signed_seat_count=signed_seat_count, salt=salt, license_management=license_management, licenses=licenses, tier=tier, remote_server_plan_start_date=remote_server_plan_start_date)
        data = billing_session.do_upgrade(upgrade_request)
        return json_success(request, data)
    except BillingError as e:
        billing_logger.warning('BillingError during upgrade: %s. remote_server=%s (%s), billing_modality=%s, schedule=%s, license_management=%s, licenses=%s', e.error_description, billing_session.remote_server.id, billing_session.remote_server.hostname, billing_modality, schedule, license_management, licenses)
        raise e
    except Exception:
        billing_logger.exception('Uncaught exception in billing:', stack_info=True)
        error_message = BillingError.CONTACT_SUPPORT.format(email=settings.ZULIP_ADMINISTRATOR)
        error_description = 'uncaught exception during upgrade'
        raise BillingError(error_description, error_message)

@zulip_login_required
@typed_endpoint
def upgrade_page(request: Union[django.http.HttpRequest, scrapy.http.Request], *, manual_license_management: bool=False, tier: Any=CustomerPlan.TIER_CLOUD_STANDARD, setup_payment_by_invoice: bool=False) -> Union[django.http.HttpResponse, str, apistar.http.Response, HttpResponseRedirect, list]:
    from corporate.lib.stripe import InitialUpgradeRequest, RealmBillingSession
    user = request.user
    assert user.is_authenticated
    if not settings.BILLING_ENABLED or user.is_guest:
        return render(request, '404.html', status=404)
    billing_modality = 'charge_automatically'
    if setup_payment_by_invoice:
        billing_modality = 'send_invoice'
    initial_upgrade_request = InitialUpgradeRequest(manual_license_management=manual_license_management, tier=tier, billing_modality=billing_modality)
    billing_session = RealmBillingSession(user)
    redirect_url, context = billing_session.get_initial_upgrade_context(initial_upgrade_request)
    if redirect_url:
        return HttpResponseRedirect(redirect_url)
    response = render(request, 'corporate/billing/upgrade.html', context=context)
    return response

@typed_endpoint
@authenticated_remote_realm_management_endpoint
def remote_realm_upgrade_page(request: Union[django.http.HttpRequest, scrapy.http.Request], billing_session: Union[str, bool, None], *, manual_license_management: bool=False, success_message: typing.Text='', tier: Any=str(CustomerPlan.TIER_SELF_HOSTED_BUSINESS) -> Union[str, settings.Settings, None, HttpResponseRedirect, list], setup_payment_by_invoice=False):
    from corporate.lib.stripe import InitialUpgradeRequest
    billing_modality = 'charge_automatically'
    if setup_payment_by_invoice:
        billing_modality = 'send_invoice'
    initial_upgrade_request = InitialUpgradeRequest(manual_license_management=manual_license_management, tier=int(tier), success_message=success_message, billing_modality=billing_modality)
    try:
        redirect_url, context = billing_session.get_initial_upgrade_context(initial_upgrade_request)
    except MissingDataError:
        return billing_session.missing_data_error_page(request)
    if redirect_url:
        return HttpResponseRedirect(redirect_url)
    response = render(request, 'corporate/billing/upgrade.html', context=context)
    return response

@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_upgrade_page(request: Union[scrapy.http.Request, django.http.HttpRequest], billing_session: Union[str, None, bool], *, manual_license_management: bool=False, success_message: typing.Text='', tier: Any=str(CustomerPlan.TIER_SELF_HOSTED_BUSINESS) -> Union[str, None, settings.Settings, HttpResponseRedirect, list], setup_payment_by_invoice=False):
    from corporate.lib.stripe import InitialUpgradeRequest
    billing_modality = 'charge_automatically'
    if setup_payment_by_invoice:
        billing_modality = 'send_invoice'
    initial_upgrade_request = InitialUpgradeRequest(manual_license_management=manual_license_management, tier=int(tier), success_message=success_message, billing_modality=billing_modality)
    try:
        redirect_url, context = billing_session.get_initial_upgrade_context(initial_upgrade_request)
    except MissingDataError:
        return billing_session.missing_data_error_page(request)
    if redirect_url:
        return HttpResponseRedirect(redirect_url)
    response = render(request, 'corporate/billing/upgrade.html', context=context)
    return response