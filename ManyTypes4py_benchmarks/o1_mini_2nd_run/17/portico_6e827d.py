from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional, Dict, Any
from urllib.parse import urlencode
import orjson
from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from pydantic import Json
from corporate.lib.decorator import (
    authenticated_remote_realm_management_endpoint,
    authenticated_remote_server_management_endpoint,
)
from corporate.models import (
    CustomerPlan,
    get_current_plan_by_customer,
    get_customer_by_realm,
)
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
def apps_view(request: HttpRequest, platform: Optional[str] = None) -> HttpResponse:
    if not settings.CORPORATE_ENABLED:
        return HttpResponseRedirect('https://zulip.com/apps/', status=301)
    return TemplateResponse(request, 'corporate/apps.html')


def app_download_link_redirect(request: HttpRequest, platform: str) -> HttpResponse:
    try:
        download_link = get_latest_github_release_download_link_for_platform(platform)
        return HttpResponseRedirect(download_link, status=302)
    except InvalidPlatformError:
        return TemplateResponse(request, '404.html', status=404)


def is_customer_on_free_trial(customer_plan: Optional[CustomerPlan]) -> bool:
    return customer_plan is not None and customer_plan.status in (
        CustomerPlan.FREE_TRIAL,
        CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL,
    )


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
def plans_view(request: HttpRequest) -> HttpResponse:
    from corporate.lib.stripe import get_free_trial_days

    realm: Optional[Realm] = get_realm_from_request(request)
    context = PlansPageContext(
        is_cloud_realm=True,
        sponsorship_url=reverse('sponsorship_request'),
        free_trial_days=get_free_trial_days(False),
        is_sponsored=realm is not None and realm.plan_type == Realm.PLAN_TYPE_STANDARD_FREE,
    )
    if is_subdomain_root_or_alias(request):
        context.sponsorship_url = f'/accounts/go/?{urlencode({"next": context.sponsorship_url})}'
    if realm is not None:
        if realm.plan_type == Realm.PLAN_TYPE_SELF_HOSTED and settings.PRODUCTION:
            return HttpResponseRedirect('https://zulip.com/plans/')
        if not request.user.is_authenticated:
            return redirect_to_login(next='/plans/')
        if request.user.is_guest:
            return TemplateResponse(request, '404.html', status=404)
        customer = get_customer_by_realm(realm)
        context.on_free_tier = customer is None and (not context.is_sponsored)
        if customer is not None:
            context.sponsorship_pending = customer.sponsorship_pending
            context.customer_plan = get_current_plan_by_customer(customer)
            if context.customer_plan is None:
                context.on_free_tier = not context.is_sponsored
            else:
                context.on_free_trial = is_customer_on_free_trial(context.customer_plan)
    context.is_new_customer = not context.on_free_tier and context.customer_plan is None and (not context.is_sponsored)
    return TemplateResponse(request, 'corporate/plans.html', context=asdict(context))


@add_google_analytics
@authenticated_remote_realm_management_endpoint
def remote_realm_plans_page(request: HttpRequest, billing_session: 'RemoteRealmBillingSession') -> HttpResponse:
    from corporate.lib.stripe import (
        get_configured_fixed_price_plan_offer,
        get_free_trial_days,
    )

    customer = billing_session.get_customer()
    context = PlansPageContext(
        is_self_hosted_realm=True,
        sponsorship_url=reverse('remote_realm_sponsorship_page', args=(billing_session.remote_realm.uuid,)),
        free_trial_days=get_free_trial_days(True),
        billing_base_url=billing_session.billing_base_url,
        is_sponsored=billing_session.is_sponsored(),
        requested_sponsorship_plan=billing_session.get_sponsorship_plan_name(customer, True),
    )
    context.on_free_tier = customer is None and (not context.is_sponsored)
    if customer is not None:
        context.sponsorship_pending = customer.sponsorship_pending
        context.customer_plan = get_current_plan_by_customer(customer)
        if customer.required_plan_tier is not None:
            configure_fixed_price_plan = get_configured_fixed_price_plan_offer(customer, customer.required_plan_tier)
            if configure_fixed_price_plan is not None:
                context.free_trial_days = None
        if context.customer_plan is None:
            context.on_free_tier = not context.is_sponsored
        else:
            context.on_free_tier = context.customer_plan.tier in (
                CustomerPlan.TIER_SELF_HOSTED_LEGACY,
                CustomerPlan.TIER_SELF_HOSTED_BASE,
            ) and (not context.is_sponsored)
            context.on_free_trial = is_customer_on_free_trial(context.customer_plan)
            if context.customer_plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
                assert context.customer_plan.end_date is not None
                context.scheduled_upgrade_plan = CustomerPlan.objects.get(
                    customer=customer,
                    billing_cycle_anchor=context.customer_plan.end_date,
                    status=CustomerPlan.NEVER_STARTED,
                )
                context.has_scheduled_upgrade = context.customer_plan.tier != context.scheduled_upgrade_plan.tier
    if billing_session.customer_plan_exists():
        context.free_trial_days = None
    context.is_new_customer = not context.on_free_tier and context.customer_plan is None and (not context.is_sponsored)
    return TemplateResponse(request, 'corporate/plans.html', context=asdict(context))


@add_google_analytics
@authenticated_remote_server_management_endpoint
def remote_server_plans_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession') -> HttpResponse:
    from corporate.lib.stripe import (
        get_configured_fixed_price_plan_offer,
        get_free_trial_days,
    )

    customer = billing_session.get_customer()
    context = PlansPageContext(
        is_self_hosted_realm=True,
        sponsorship_url=reverse('remote_server_sponsorship_page', args=(billing_session.remote_server.uuid,)),
        free_trial_days=get_free_trial_days(True),
        billing_base_url=billing_session.billing_base_url,
        is_sponsored=billing_session.is_sponsored(),
        requested_sponsorship_plan=billing_session.get_sponsorship_plan_name(customer, True),
    )
    context.on_free_tier = customer is None and (not context.is_sponsored)
    if customer is not None:
        context.sponsorship_pending = customer.sponsorship_pending
        context.customer_plan = get_current_plan_by_customer(customer)
        if customer.required_plan_tier is not None:
            configure_fixed_price_plan = get_configured_fixed_price_plan_offer(customer, customer.required_plan_tier)
            if configure_fixed_price_plan is not None:
                context.free_trial_days = None
        if context.customer_plan is None:
            context.on_free_tier = not context.is_sponsored
        else:
            context.on_free_tier = context.customer_plan.tier in (
                CustomerPlan.TIER_SELF_HOSTED_LEGACY,
                CustomerPlan.TIER_SELF_HOSTED_BASE,
            )
            context.on_free_trial = is_customer_on_free_trial(context.customer_plan)
            if context.customer_plan.status == CustomerPlan.SWITCH_PLAN_TIER_AT_PLAN_END:
                assert context.customer_plan.end_date is not None
                context.scheduled_upgrade_plan = CustomerPlan.objects.get(
                    customer=customer,
                    billing_cycle_anchor=context.customer_plan.end_date,
                    status=CustomerPlan.NEVER_STARTED,
                )
                context.has_scheduled_upgrade = context.customer_plan.tier != context.scheduled_upgrade_plan.tier
        if billing_session.customer_plan_exists():
            context.free_trial_days = None
    context.is_new_customer = not context.on_free_tier and context.customer_plan is None and (not context.is_sponsored)
    return TemplateResponse(request, 'corporate/plans.html', context=asdict(context))


@add_google_analytics
def team_view(request: HttpRequest) -> HttpResponse:
    if not settings.ZILENCER_ENABLED:
        return HttpResponseRedirect('https://zulip.com/team/', status=301)
    try:
        with open(settings.CONTRIBUTOR_DATA_FILE_PATH, 'rb') as f:
            data: Dict[str, Any] = orjson.loads(f.read())
    except FileNotFoundError:
        data = {'contributors': [], 'date': 'Never ran.'}
    return TemplateResponse(
        request,
        'corporate/team.html',
        context={
            'page_params': {'page_type': 'team', 'contributors': data['contributors']},
            'date': data['date'],
        },
    )


@add_google_analytics
def landing_view(request: HttpRequest, template_name: str) -> HttpResponse:
    context: Dict[str, Any] = latest_info_context()
    context.update(
        {
            'billing_base_url': '',
            'tier_cloud_standard': str(CustomerPlan.TIER_CLOUD_STANDARD),
            'tier_cloud_plus': str(CustomerPlan.TIER_CLOUD_PLUS),
        }
    )
    return TemplateResponse(request, template_name, context)


@add_google_analytics
def hello_view(request: HttpRequest) -> HttpResponse:
    return TemplateResponse(request, 'corporate/hello.html', latest_info_context())


@add_google_analytics
def communities_view(request: HttpRequest) -> HttpResponse:
    eligible_realms: list[Dict[str, Any]] = []
    unique_org_type_ids: set[int] = set()
    want_to_be_advertised_realms = Realm.objects.filter(
        want_advertise_in_communities_directory=True
    ).exclude(description='').order_by('name')
    for realm in want_to_be_advertised_realms:
        open_to_public = (
            not realm.invite_required and (not realm.emails_restricted_to_domains)
        )
        if realm.allow_web_public_streams_access() or open_to_public:
            org_type = next(
                (
                    org_type_key
                    for org_type_key in Realm.ORG_TYPES
                    if Realm.ORG_TYPES[org_type_key]['id'] == realm.org_type
                ),
                None,
            )
            if org_type is not None:
                eligible_realms.append(
                    {
                        'id': realm.id,
                        'name': realm.name,
                        'realm_url': realm.url,
                        'logo_url': get_realm_icon_url(realm),
                        'description': get_realm_text_description(realm),
                        'org_type_key': org_type,
                    }
                )
                unique_org_type_ids.add(realm.org_type)
    CATEGORIES_TO_OFFER = ['opensource', 'research', 'community']
    org_types: Dict[str, Any] = {}
    for org_type in CATEGORIES_TO_OFFER:
        if Realm.ORG_TYPES[org_type]['id'] in unique_org_type_ids:
            org_types[org_type] = Realm.ORG_TYPES[org_type]
    return TemplateResponse(
        request,
        'corporate/communities.html',
        context={'eligible_realms': eligible_realms, 'org_types': org_types},
    )


@zulip_login_required
def invoices_page(request: HttpRequest) -> HttpResponse:
    from corporate.lib.stripe import RealmBillingSession

    user = request.user
    assert user.is_authenticated
    if not user.has_billing_access:
        return HttpResponseRedirect(reverse('billing_page'))
    billing_session = RealmBillingSession(user=user, realm=user.realm)
    list_invoices_session_url: str = billing_session.get_past_invoices_session_url()
    return HttpResponseRedirect(list_invoices_session_url)


@authenticated_remote_realm_management_endpoint
def remote_realm_invoices_page(request: HttpRequest, billing_session: 'RemoteRealmBillingSession') -> HttpResponse:
    list_invoices_session_url: str = billing_session.get_past_invoices_session_url()
    return HttpResponseRedirect(list_invoices_session_url)


@authenticated_remote_server_management_endpoint
def remote_server_invoices_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession') -> HttpResponse:
    list_invoices_session_url: str = billing_session.get_past_invoices_session_url()
    return HttpResponseRedirect(list_invoices_session_url)


@zulip_login_required
@typed_endpoint
def customer_portal(
    request: HttpRequest,
    *,
    return_to_billing_page: bool = False,
    manual_license_management: bool = False,
    tier: Optional[str] = None,
    setup_payment_by_invoice: bool = False,
) -> HttpResponse:
    from corporate.lib.stripe import RealmBillingSession

    user = request.user
    assert user.is_authenticated
    if not user.has_billing_access:
        return HttpResponseRedirect(reverse('billing_page'))
    billing_session = RealmBillingSession(user=user, realm=user.realm)
    review_billing_information_url: str = billing_session.get_stripe_customer_portal_url(
        return_to_billing_page, manual_license_management, tier, setup_payment_by_invoice
    )
    return HttpResponseRedirect(review_billing_information_url)


@typed_endpoint
@authenticated_remote_realm_management_endpoint
def remote_realm_customer_portal(
    request: HttpRequest,
    billing_session: 'RemoteRealmBillingSession',
    *,
    return_to_billing_page: bool = False,
    manual_license_management: bool = False,
    tier: Optional[str] = None,
    setup_payment_by_invoice: bool = False,
) -> HttpResponse:
    review_billing_information_url: str = billing_session.get_stripe_customer_portal_url(
        return_to_billing_page, manual_license_management, tier, setup_payment_by_invoice
    )
    return HttpResponseRedirect(review_billing_information_url)


@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_customer_portal(
    request: HttpRequest,
    billing_session: 'RemoteServerBillingSession',
    *,
    return_to_billing_page: bool = False,
    manual_license_management: bool = False,
    tier: Optional[str] = None,
    setup_payment_by_invoice: bool = False,
) -> HttpResponse:
    review_billing_information_url: str = billing_session.get_stripe_customer_portal_url(
        return_to_billing_page, manual_license_management, tier, setup_payment_by_invoice
    )
    return HttpResponseRedirect(review_billing_information_url)
