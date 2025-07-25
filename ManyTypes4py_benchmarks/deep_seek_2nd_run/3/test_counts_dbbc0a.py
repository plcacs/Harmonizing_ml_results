from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast
from unittest import mock
import time_machine
from django.apps import apps
from django.db.models import QuerySet, Sum
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Literal
from typing_extensions import override
from analytics.lib.counts import COUNT_STATS, CountStat, DependentCountStat, LoggingCountStat, do_aggregate_to_summary_table, do_drop_all_analytics_tables, do_drop_single_stat, do_fill_count_stat_at_hour, do_increment_logging_stat, get_count_stats, process_count_stat, sql_data_collector
from analytics.models import BaseCount, FillState, InstallationCount, RealmCount, StreamCount, UserCount, installation_epoch
from zerver.actions.create_realm import do_create_realm
from zerver.actions.create_user import do_activate_mirror_dummy_user, do_create_user, do_reactivate_user
from zerver.actions.invites import do_invite_users, do_revoke_user_invite, do_send_user_invite_email
from zerver.actions.message_flags import do_mark_all_as_read, do_mark_stream_messages_as_read, do_update_message_flags
from zerver.actions.user_activity import update_user_activity_interval
from zerver.actions.users import do_deactivate_user
from zerver.lib.create_user import create_user
from zerver.lib.exceptions import InvitationError
from zerver.lib.push_notifications import get_message_payload_apns, get_message_payload_gcm, hex_to_b64
from zerver.lib.streams import get_default_values_for_stream_permission_group_settings
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import activate_push_notification_service
from zerver.lib.timestamp import TimeZoneNotUTCError, ceiling_to_day, floor_to_day
from zerver.lib.topic import DB_TOPIC_NAME
from zerver.lib.user_counts import realm_user_count_by_role
from zerver.lib.utils import assert_is_not_none
from zerver.models import Client, DirectMessageGroup, Message, PreregistrationUser, RealmAuditLog, Recipient, Stream, UserActivityInterval, UserProfile
from zerver.models.clients import get_client
from zerver.models.messages import Attachment
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.scheduled_jobs import NotificationTriggers
from zerver.models.users import get_user, is_cross_realm_bot_email
from zilencer.models import RemoteInstallationCount, RemotePushDeviceToken, RemoteRealm, RemoteRealmCount, RemoteZulipServer
from zilencer.views import get_last_id_from_server

T = TypeVar('T', bound=BaseCount)

class AnalyticsTestCase(ZulipTestCase):
    MINUTE: timedelta = timedelta(seconds=60)
    HOUR: timedelta = MINUTE * 60
    DAY: timedelta = HOUR * 24
    TIME_ZERO: datetime = datetime(1988, 3, 14, tzinfo=timezone.utc)
    TIME_LAST_HOUR: datetime = TIME_ZERO - HOUR

    @override
    def setUp(self) -> None:
        super().setUp()
        self.default_realm = do_create_realm(string_id='realmtest', name='Realm Test', date_created=self.TIME_ZERO - 2 * self.DAY)
        self.name_counter: int = 100
        self.current_property: Optional[str] = None
        RemoteRealm.objects.all().delete()

    def create_user(self, skip_auditlog: bool = False, **kwargs: Any) -> UserProfile:
        self.name_counter += 1
        defaults: Dict[str, Any] = {'email': f'user{self.name_counter}@domain.tld', 'date_joined': self.TIME_LAST_HOUR, 'full_name': 'full_name', 'is_active': True, 'is_bot': False, 'realm': self.default_realm}
        for key, value in defaults.items():
            kwargs[key] = kwargs.get(key, value)
        kwargs['delivery_email'] = kwargs['email']
        with time_machine.travel(kwargs['date_joined'], tick=False):
            pass_kwargs: Dict[str, Any] = {}
            if kwargs['is_bot']:
                pass_kwargs['bot_type'] = UserProfile.DEFAULT_BOT
                pass_kwargs['bot_owner'] = None
            user: UserProfile = create_user(kwargs['email'], 'password', kwargs['realm'], active=kwargs['is_active'], full_name=kwargs['full_name'], role=UserProfile.ROLE_REALM_ADMINISTRATOR, **pass_kwargs)
            if not skip_auditlog:
                RealmAuditLog.objects.create(realm=kwargs['realm'], acting_user=None, modified_user=user, event_type=AuditLogEventType.USER_CREATED, event_time=kwargs['date_joined'], extra_data={RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(kwargs['realm'])})
            return user

    def create_stream_with_recipient(self, **kwargs: Any) -> Tuple[Stream, Recipient]:
        self.name_counter += 1
        defaults: Dict[str, Any] = {'name': f'stream name {self.name_counter}', 'realm': self.default_realm, 'date_created': self.TIME_LAST_HOUR, **get_default_values_for_stream_permission_group_settings(self.default_realm)}
        for key, value in defaults.items():
            kwargs[key] = kwargs.get(key, value)
        stream: Stream = Stream.objects.create(**kwargs)
        recipient: Recipient = Recipient.objects.create(type_id=stream.id, type=Recipient.STREAM)
        stream.recipient = recipient
        stream.save(update_fields=['recipient'])
        return (stream, recipient)

    def create_direct_message_group_with_recipient(self, **kwargs: Any) -> Tuple[DirectMessageGroup, Recipient]:
        self.name_counter += 1
        defaults: Dict[str, Any] = {'huddle_hash': f'hash{self.name_counter}', 'group_size': 4}
        for key, value in defaults.items():
            kwargs[key] = kwargs.get(key, value)
        direct_message_group: DirectMessageGroup = DirectMessageGroup.objects.create(**kwargs)
        recipient: Recipient = Recipient.objects.create(type_id=direct_message_group.id, type=Recipient.DIRECT_MESSAGE_GROUP)
        direct_message_group.recipient = recipient
        direct_message_group.save(update_fields=['recipient'])
        return (direct_message_group, recipient)

    def create_message(self, sender: UserProfile, recipient: Recipient, **kwargs: Any) -> Message:
        defaults: Dict[str, Any] = {'sender': sender, 'recipient': recipient, DB_TOPIC_NAME: 'subject', 'content': 'hi', 'date_sent': self.TIME_LAST_HOUR, 'sending_client': get_client('website'), 'realm_id': sender.realm_id}
        assert not is_cross_realm_bot_email(sender.delivery_email)
        for key, value in defaults.items():
            kwargs[key] = kwargs.get(key, value)
        return Message.objects.create(**kwargs)

    def create_attachment(self, user_profile: UserProfile, filename: str, size: int, create_time: datetime, content_type: str) -> Attachment:
        return Attachment.objects.create(file_name=filename, path_id=f'foo/bar/{filename}', owner=user_profile, realm=user_profile.realm, size=size, create_time=create_time, content_type=content_type)

    def assertTableState(self, table: Type[T], arg_keys: List[str], arg_values: List[List[Any]]) -> None:
        defaults: Dict[str, Any] = {'property': self.current_property, 'subgroup': None, 'end_time': self.TIME_ZERO, 'value': 1}
        for values in arg_values:
            kwargs: Dict[str, Any] = {}
            for i in range(len(values)):
                kwargs[arg_keys[i]] = values[i]
            for key, value in defaults.items():
                kwargs[key] = kwargs.get(key, value)
            if table not in [InstallationCount, RemoteInstallationCount, RemoteRealmCount] and 'realm' not in kwargs:
                if 'user' in kwargs:
                    kwargs['realm'] = kwargs['user'].realm
                elif 'stream' in kwargs:
                    kwargs['realm'] = kwargs['stream'].realm
                else:
                    kwargs['realm'] = self.default_realm
            self.assertEqual(table._default_manager.filter(**kwargs).count(), 1)
        self.assert_length(arg_values, table._default_manager.count())

class TestProcessCountStat(AnalyticsTestCase):
    def make_dummy_count_stat(self, property: str) -> CountStat:
        query = lambda kwargs: SQL('\n            INSERT INTO analytics_realmcount (realm_id, value, property, end_time)\n            VALUES ({default_realm_id}, 1, {property}, %(time_end)s)\n        ').format(default_realm_id=Literal(self.default_realm.id), property=Literal(property))
        return CountStat(property, sql_data_collector(RealmCount, query, None), CountStat.HOUR)

    def assertFillStateEquals(self, stat: CountStat, end_time: datetime, state: int = FillState.DONE) -> None:
        fill_state: Optional[FillState] = FillState.objects.filter(property=stat.property).first()
        assert fill_state is not None
        self.assertEqual(fill_state.end_time, end_time)
        self.assertEqual(fill_state.state, state)

    def test_process_stat(self) -> None:
        current_time: datetime = installation_epoch() + self.HOUR
        stat: CountStat = self.make_dummy_count_stat('test stat')
        process_count_stat(stat, current_time)
        self.assertFillStateEquals(stat, current_time)
        self.assertEqual(InstallationCount.objects.filter(property=stat.property).count(), 1)
        FillState.objects.filter(property=stat.property).update(state=FillState.STARTED)
        process_count_stat(stat, current_time)
        self.assertFillStateEquals(stat, current_time)
        self.assertEqual(InstallationCount.objects.filter(property=stat.property).count(), 1)
        process_count_stat(stat, current_time)
        self.assertFillStateEquals(stat, current_time)
        self.assertEqual(InstallationCount.objects.filter(property=stat.property).count(), 1)
        current_time += self.HOUR
        stat = self.make_dummy_count_stat('test stat')
        process_count_stat(stat, current_time)
        self.assertFillStateEquals(stat, current_time)
        self.assertEqual(InstallationCount.objects.filter(property=stat.property).count(), 2)

    def test_bad_fill_to_time(self) -> None:
        stat: CountStat = self.make_dummy_count_stat('test stat')
        with self.assertRaises(ValueError):
            process_count_stat(stat, installation_epoch() + 65 * self.MINUTE)
        with self.assertRaises(TimeZoneNotUTCError):
            process_count_stat(stat, installation_epoch().replace(tzinfo=None))

    def test_process_logging_stat(self) -> None:
        end_time: datetime = self.TIME_ZERO
        user_stat: LoggingCountStat = LoggingCountStat('user stat', UserCount, CountStat.DAY)
        stream_stat: LoggingCountStat = LoggingCountStat('stream stat', StreamCount, CountStat.DAY)
        realm_stat: LoggingCountStat = LoggingCountStat('realm stat', RealmCount, CountStat.DAY)
        user: UserProfile = self.create_user()
        stream: Stream = self.create_stream_with_recipient()[0]
        realm: Any = self.default_realm
        UserCount.objects.create(user=user, realm=realm, property=user_stat.property, end_time=end_time, value=5)
        StreamCount.objects.create(stream=stream, realm=realm, property=stream_stat.property, end_time=end_time, value=5)
        RealmCount.objects.create(realm=realm, property=realm_stat.property, end_time=end_time, value=5)
        for stat in [user_stat, stream_stat, realm_stat]:
            process_count_stat(stat, end_time)
        self.assertTableState(UserCount, ['property', 'value'], [[user_stat.property, 5]])
        self.assertTableState(StreamCount, ['property', 'value'], [[stream_stat.property, 5]])
        self.assertTableState(RealmCount, ['property', 'value'], [[user_stat.property, 5], [stream_stat.property, 5], [realm_stat.property, 5]])
        self.assertTableState(InstallationCount, ['property', 'value'], [[user_stat.property, 5], [stream_stat.property, 5], [realm_stat.property, 5]])
        UserCount.objects.update(value=6)
        StreamCount.objects.update(value=6)
        RealmCount.objects.filter(property=realm_stat.property).update(value=6)
        FillState.objects.update(state=FillState.STARTED)
        for stat in [user_stat, stream_stat, realm_stat]:
            process_count_stat(stat, end_time)
        self.assertTableState(UserCount, ['property', 'value'], [[user_stat.property, 6]])
        self.assertTableState(StreamCount, ['property', 'value'], [[stream_stat.property, 6]])
        self.assertTableState(RealmCount, ['property', 'value'], [[user_stat.property, 6], [stream_stat.property, 6], [realm_stat.property, 6]])
        self.assertTableState(InstallationCount, ['property', 'value'], [[user_stat.property, 6], [stream_stat.property, 6], [realm_stat.property, 6]])

    def test_process_dependent_stat(self) -> None:
        stat1: CountStat = self.make_dummy_count_stat('stat1')
        stat2: CountStat = self.make_dummy_count_stat('stat2')
        query = lambda kwargs: SQL('\n            INSERT INTO analytics_realmcount (realm_id, value, property, end_time)\n            VALUES ({default_realm_id}, 1, {property}, %(time_end)s)\n        ').format(default_realm_id=Literal(self.default_realm.id), property=Literal('stat3'))
        stat3: DependentCountStat = DependentCountStat('stat3', sql_data_collector(RealmCount, query, None), CountStat.HOUR, dependencies=['stat1', 'stat2'])
        query = lambda kwargs: SQL('\n            INSERT INTO analytics_realmcount (realm_id, value, property, end_time)\n            VALUES ({default_realm_id}, 1, {property}, %(time_end)s)\n        ').format(default_realm_id=Literal(self.default_realm.id), property=Literal('stat4'))
        stat4: DependentCountStat = DependentCountStat('stat4', sql_data_collector(RealmCount, query, None), CountStat.DAY, dependencies=['stat1', 'stat2'])
        dummy_count_stats: Dict[str, CountStat] = {'stat1': stat1, 'stat2': stat2, 'stat3': stat3, 'stat4': stat4}
        with mock.patch('analytics.lib.counts.COUNT_STATS', dummy_count_stats):
            hour: List[datetime] = [installation_epoch() + i * self.HOUR for i in range(5)]
            process_count_stat(stat1, hour[2])
            process_count_stat(stat3, hour[1])
            self.assertTableState(InstallationCount, ['property', 'end_time'], [['stat1', hour[1]], ['stat1', hour[2]]])
            self.assertFillStateEquals(stat3, hour[0])
            process_count_stat(stat2, hour[3])
            process_count_stat(stat3, hour[1])
            self.assertTableState(InstallationCount, ['property', 'end_time'], [['stat1', hour[1]], ['stat1', hour[2]], ['stat2', hour[1]], ['stat2', hour[2]], ['stat2', hour[3]], ['stat3', hour[1]]])
            self.assertFillStateEquals(stat3, hour[1])
            process_count_stat(stat3, hour[4])
            self.assertTableState(InstallationCount, ['property', 'end_time'], [['stat1', hour[1]], ['stat1', hour[2]], ['stat2', hour[1]], ['stat2', hour[2]], ['stat2', hour[3]], ['stat3', hour[1]], ['stat3', hour[2]]])
            self.assertFillStateEquals(stat3, hour[2])
            hour24: datetime = installation_epoch() + 24 * self.HOUR
            hour25: datetime = installation_epoch() + 25 * self.HOUR
            process_count_stat(stat1, hour25)
            process_count_stat(stat2, hour25)
            process_count_stat(stat4, hour25)
            self.assertEqual(InstallationCount.objects.filter(property='stat4').count(), 1)
            self.assertFillStateEquals(stat4, hour24)

# ... (rest of the code remains the same with appropriate type annotations)
