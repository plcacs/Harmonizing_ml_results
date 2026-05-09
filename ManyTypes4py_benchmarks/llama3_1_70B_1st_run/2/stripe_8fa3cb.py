from typing import Optional, Tuple

def get_push_status_for_remote_request(remote_server: Optional['RemoteZulipServer'], 
                                      remote_realm: Optional['RemoteRealm']) -> 'PushNotificationsEnabledStatus':
    customer: Optional['Customer'] = None
    current_plan: Optional['CustomerPlan'] = None
    realm_billing_session: Optional['RemoteRealmBillingSession'] = None
    server_billing_session: Optional['RemoteServerBillingSession'] = None
    if remote_realm is not None:
        realm_billing_session = RemoteRealmBillingSession(remote_realm)
        if realm_billing_session.is_sponsored():
            return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Community plan')
        customer = realm_billing_session.get_customer()
        if customer is not None:
            current_plan = get_current_plan_by_customer(customer)
    if customer is None or current_plan is None:
        server_billing_session = RemoteServerBillingSession(remote_server)
        if server_billing_session.is_sponsored():
            return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Community plan')
        customer = server_billing_session.get_customer()
        if customer is not None:
            current_plan = get_current_plan_by_customer(customer)
    if realm_billing_session is not None:
        user_count_billing_session = realm_billing_session
    else:
        assert server_billing_session is not None
        user_count_billing_session = server_billing_session
    user_count: Optional[int] = None
    if current_plan is None:
        try:
            user_count = user_count_billing_session.current_count_for_billed_licenses()
        except MissingDataError:
            return PushNotificationsEnabledStatus(can_push=False, expected_end_timestamp=None, message='Missing data')
        if user_count > MAX_USERS_WITHOUT_PLAN:
            return PushNotificationsEnabledStatus(can_push=False, expected_end_timestamp=None, message='Push notifications access with 10+ users requires signing up for a plan. https://zulip.com/plans/')
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='No plan few users')
    if current_plan.status not in [CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL]:
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Active plan')
    try:
        user_count = user_count_billing_session.current_count_for_billed_licenses()
    except MissingDataError:
        user_count = None
    if user_count is not None and user_count <= MAX_USERS_WITHOUT_PLAN:
        return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=None, message='Expiring plan few users')
    expected_end_timestamp: int = datetime_to_timestamp(user_count_billing_session.get_next_billing_cycle(current_plan))
    return PushNotificationsEnabledStatus(can_push=True, expected_end_timestamp=expected_end_timestamp, message='Scheduled end')
