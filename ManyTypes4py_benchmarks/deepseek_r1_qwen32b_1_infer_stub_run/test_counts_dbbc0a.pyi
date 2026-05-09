from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from unittest import mock
from django.db.models import Sum
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Literal
from typing_extensions import override
from analytics.lib.counts import COUNT_STATS, CountStat, DependentCountStat, LoggingCountStat
from analytics.models import BaseCount, FillState, InstallationCount, RealmCount, StreamCount, UserCount
from zerver.actions.create_realm import do_create_realm
from zerver.actions.create_user import do_activate_mirror_dummy_user, do_create_user, do_reactivate_user
from zerver.actions.invites import do_invite_users, do_revoke_user_invite, do_send_user_invite_email
from zerver.actions.message_flags import do_mark_all_as_read, do_mark_stream_messages_as_read, do_update_message_flags
from zerver.actions.user_activity import update_user_activity_interval
from zerver.actions.users import do_deactivate_user
from zerver.lib.test_classes import ZulipTestCase
from zerver.models import Client, DirectMessageGroup, Message, PreregistrationUser, RealmAuditLog, Recipient, Stream, UserActivityInterval, UserProfile
from zilencer.models import RemoteInstallationCount, RemotePushDeviceToken, RemoteRealm, RemoteRealmCount, RemoteZulipServer

class AnalyticsTestCase(ZulipTestCase):
    MINUTE: timedelta
    HOUR: timedelta
    DAY: timedelta
    TIME_ZERO: datetime
    TIME_LAST_HOUR: datetime

    def setUp(self) -> None:
        ...

    def create_user(self, skip_auditlog: bool = False, **kwargs: Any) -> UserProfile:
        ...

    def create_stream_with_recipient(self, **kwargs: Any) -> tuple[Stream, Recipient]:
        ...

    def create_direct_message_group_with_recipient(self, **kwargs: Any) -> tuple[DirectMessageGroup, Recipient]:
        ...

    def create_message(self, sender: UserProfile, recipient: Recipient, **kwargs: Any) -> Message:
        ...

    def create_attachment(self, user_profile: UserProfile, filename: str, size: int, create_time: datetime, content_type: str) -> Attachment:
        ...

    def assertTableState(self, table: type[BaseCount], arg_keys: List[str], arg_values: List[List[Any]]) -> None:
        ...

class TestProcessCountStat(AnalyticsTestCase):
    def make_dummy_count_stat(self, property: str) -> CountStat:
        ...

    def assertFillStateEquals(self, stat: CountStat, end_time: datetime, state: FillState = FillState.DONE) -> None:
        ...

    def test_process_stat(self) -> None:
        ...

    def test_bad_fill_to_time(self) -> None:
        ...

    def test_process_logging_stat(self) -> None:
        ...

    def test_process_dependent_stat(self) -> None:
        ...

class TestCountStats(AnalyticsTestCase):
    @override
    def setUp(self) -> None:
        ...

    def test_upload_quota_used_bytes(self) -> None:
        ...

    def test_messages_sent_by_is_bot(self) -> None:
        ...

    def test_messages_sent_by_is_bot_realm_constraint(self) -> None:
        ...

    def test_messages_sent_by_message_type(self) -> None:
        ...

    def test_messages_sent_by_message_type_realm_constraint(self) -> None:
        ...

    def test_messages_sent_to_recipients_with_same_id(self) -> None:
        ...

    def test_messages_sent_by_client(self) -> None:
        ...

    def test_messages_sent_by_client_realm_constraint(self) -> None:
        ...

    def test_messages_sent_to_stream_by_is_bot(self) -> None:
        ...

    def test_messages_sent_to_stream_by_is_bot_realm_constraint(self) -> None:
        ...

    def create_interval(self, user: UserProfile, start_offset: timedelta, end_offset: timedelta) -> None:
        ...

    def test_1day_actives(self) -> None:
        ...

    def test_1day_actives_realm_constraint(self) -> None:
        ...

    def test_15day_actives(self) -> None:
        ...

    def test_15day_actives_realm_constraint(self) -> None:
        ...

    def test_minutes_active(self) -> None:
        ...

    def test_minutes_active_realm_constraint(self) -> None:
        ...

    def test_last_successful_fill(self) -> None:
        ...

class TestDoAggregateToSummaryTable(AnalyticsTestCase):
    def test_no_aggregated_zeros(self) -> None:
        ...

class TestDoIncrementLoggingStat(AnalyticsTestCase):
    def test_table_and_id_args(self) -> None:
        ...

    def test_frequency(self) -> None:
        ...

    def test_get_or_create(self) -> None:
        ...

    def test_increment(self) -> None:
        ...

    def test_do_increment_logging_start_query_count(self) -> None:
        ...

class TestLoggingCountStats(AnalyticsTestCase):
    def test_aggregation(self) -> None:
        ...

    @activate_push_notification_service()
    def test_mobile_pushes_received_count(self) -> None:
        ...

    def test_invites_sent(self) -> None:
        ...

    def test_messages_read_hour(self) -> None:
        ...

class TestDeleteStats(AnalyticsTestCase):
    def test_do_drop_all_analytics_tables(self) -> None:
        ...

    def test_do_drop_single_stat(self) -> None:
        ...

class TestActiveUsersAudit(AnalyticsTestCase):
    @override
    def setUp(self) -> None:
        ...

    def add_event(self, event_type: RealmAuditLog.EventType, days_offset: float, user: Optional[UserProfile] = None) -> None:
        ...

    def test_user_deactivated_in_future(self) -> None:
        ...

    def test_user_reactivated_in_future(self) -> None:
        ...

    def test_user_active_then_deactivated_same_day(self) -> None:
        ...

    def test_user_inactive_then_activated_same_day(self) -> None:
        ...

    def test_user_active_then_deactivated_with_day_gap(self) -> None:
        ...

    def test_user_deactivated_then_reactivated_with_day_gap(self) -> None:
        ...

    def test_event_types(self) -> None:
        ...

    def test_multiple_users_realms_and_bots(self) -> None:
        ...

    def test_update_from_two_days_ago(self) -> None:
        ...

    def test_empty_realm_or_user_with_no_relevant_activity(self) -> None:
        ...

    def test_max_audit_entry_is_unrelated(self) -> None:
        ...

    def test_simultaneous_unrelated_audit_entry(self) -> None:
        ...

    def test_simultaneous_max_audit_entries_of_different_users(self) -> None:
        ...

    def test_end_to_end_with_actions_dot_py(self) -> None:
        ...

class TestRealmActiveHumans(AnalyticsTestCase):
    @override
    def setUp(self) -> None:
        ...

    def mark_15day_active(self, user: UserProfile, end_time: Optional[datetime] = None) -> None:
        ...

    def test_basic_logic(self) -> None:
        ...

    def test_bots_not_counted(self) -> None:
        ...

    def test_multiple_users_realms_and_times(self) -> None:
        ...

    def test_end_to_end(self) -> None:
        ...

def get_last_id_from_server_ignores_null() -> None:
    ...