from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, ContextManager, Dict, List, Tuple, Type, Optional
from unittest import mock

import time_machine
from django.apps import apps
from django.db.models import Sum, Model, QuerySet
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Literal
from typing_extensions import override

from analytics.lib.counts import (
    COUNT_STATS,
    CountStat,
    DependentCountStat,
    LoggingCountStat,
    do_aggregate_to_summary_table,
    do_drop_all_analytics_tables,
    do_drop_single_stat,
    do_fill_count_stat_at_hour,
    do_increment_logging_stat,
    get_count_stats,
    process_count_stat,
    sql_data_collector,
)
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
        defaults: Dict[str, Any] = {
            'email': f'user{self.name_counter}@domain.tld',
            'date_joined': self.TIME_LAST_HOUR,
            'full_name': 'full_name',
            'is_active': True,
            'is_bot': False,
            'realm': self.default_realm,
        }
        for key, value in defaults.items():
            kwargs[key] = kwargs.get(key, value)
        kwargs['delivery_email'] = kwargs['email']
        with time_machine.travel(kwargs['date_joined'], tick=False):
            pass
        pass_kwargs: Dict[str, Any] = {}
        if kwargs['is_bot']:
            pass_kwargs['bot_type'] = UserProfile.DEFAULT_BOT
            pass_kwargs['bot_owner'] = None
        user: UserProfile = create_user(
            kwargs['email'], 'password', kwargs['realm'], active=kwargs['is_active'], full_name=kwargs['full_name'], role=UserProfile.ROLE_REALM_ADMINISTRATOR, **pass_kwargs
        )
        if not skip_auditlog:
            RealmAuditLog.objects.create(
                realm=kwargs['realm'],
                acting_user=None,
                modified_user=user,
                event_type=AuditLogEventType.USER_CREATED,
                event_time=kwargs['date_joined'],
                extra_data={RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(kwargs['realm'])},
            )
        return user

    def create_stream_with_recipient(self, **kwargs: Any) -> Tuple[Stream, Recipient]:
        self.name_counter += 1
        defaults: Dict[str, Any] = {
            'name': f'stream name {self.name_counter}',
            'realm': self.default_realm,
            'date_created': self.TIME_LAST_HOUR,
            **get_default_values_for_stream_permission_group_settings(self.default_realm),
        }
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
        defaults: Dict[str, Any] = {
            'sender': sender,
            'recipient': recipient,
            DB_TOPIC_NAME: 'subject',
            'content': 'hi',
            'date_sent': self.TIME_LAST_HOUR,
            'sending_client': get_client('website'),
            'realm_id': sender.realm_id,
        }
        assert not is_cross_realm_bot_email(sender.delivery_email)
        for key, value in defaults.items():
            kwargs[key] = kwargs.get(key, value)
        return Message.objects.create(**kwargs)

    def create_attachment(self, user_profile: UserProfile, filename: str, size: int, create_time: datetime, content_type: str) -> Attachment:
        return Attachment.objects.create(
            file_name=filename,
            path_id=f'foo/bar/{filename}',
            owner=user_profile,
            realm=user_profile.realm,
            size=size,
            create_time=create_time,
            content_type=content_type,
        )

    def assertTableState(self, table: Type[Model], arg_keys: List[str], arg_values: List[List[Any]]) -> None:
        """Assert that the state of a *Count table is what it should be.

        Example usage:
            self.assertTableState(RealmCount, ['property', 'subgroup', 'realm'],
                                  [['p1', 4], ['p2', 10, self.alt_realm]])

        table -- A *Count table.
        arg_keys -- List of columns of <table>.
        arg_values -- List of "rows" of <table>.
            Each entry of arg_values (e.g. ['p1', 4]) represents a row of <table>.
            The i'th value of the entry corresponds to the i'th arg_key, so e.g.
            the first arg_values entry here corresponds to a row of RealmCount
            with property='p1' and subgroup=10.
            Any columns not specified (in this case, every column of RealmCount
            other than property and subgroup) are either set to default values,
            or are ignored.

        The function checks that every entry of arg_values matches exactly one
        row of <table>, and that no additional rows exist. Note that this means
        checking a table with duplicate rows is not supported.
        """
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
        query: Callable[[Dict[str, Any]], SQL] = lambda kwargs: SQL(
            '\n            INSERT INTO analytics_realmcount (realm_id, value, property, end_time)\n            VALUES ({default_realm_id}, 1, {property}, %(time_end)s)\n        '
        ).format(default_realm_id=Literal(self.default_realm.id), property=Literal(property))
        return CountStat(property, sql_data_collector(RealmCount, query, None), CountStat.HOUR)

    def assertFillStateEquals(self, stat: CountStat, end_time: datetime, state: str = FillState.DONE) -> None:
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
        realm = self.default_realm
        UserCount.objects.create(user=user, realm=realm, property=user_stat.property, end_time=end_time, value=5)
        StreamCount.objects.create(stream=stream, realm=realm, property=stream_stat.property, end_time=end_time, value=5)
        RealmCount.objects.create(realm=realm, property=realm_stat.property, end_time=end_time, value=5)
        for stat_item in [user_stat, stream_stat, realm_stat]:
            process_count_stat(stat_item, end_time)
        self.assertTableState(UserCount, ['property', 'value'], [[user_stat.property, 5]])
        self.assertTableState(StreamCount, ['property', 'value'], [[stream_stat.property, 5]])
        self.assertTableState(RealmCount, ['property', 'value'], [[user_stat.property, 5], [stream_stat.property, 5], [realm_stat.property, 5]])
        self.assertTableState(InstallationCount, ['property', 'value'], [[user_stat.property, 5], [stream_stat.property, 5], [realm_stat.property, 5]])
        UserCount.objects.update(value=6)
        StreamCount.objects.update(value=6)
        RealmCount.objects.filter(property=realm_stat.property).update(value=6)
        FillState.objects.update(state=FillState.STARTED)
        for stat_item in [user_stat, stream_stat, realm_stat]:
            process_count_stat(stat_item, end_time)
        self.assertTableState(UserCount, ['property', 'value'], [[user_stat.property, 6]])
        self.assertTableState(StreamCount, ['property', 'value'], [[stream_stat.property, 6]])
        self.assertTableState(RealmCount, ['property', 'value'], [[user_stat.property, 6], [stream_stat.property, 6], [realm_stat.property, 6]])
        self.assertTableState(InstallationCount, ['property', 'value'], [[user_stat.property, 6], [stream_stat.property, 6], [realm_stat.property, 6]])

    def test_process_dependent_stat(self) -> None:
        stat1: CountStat = self.make_dummy_count_stat('stat1')
        stat2: CountStat = self.make_dummy_count_stat('stat2')
        query: Callable[[Dict[str, Any]], SQL] = lambda kwargs: SQL(
            '\n            INSERT INTO analytics_realmcount (realm_id, value, property, end_time)\n            VALUES ({default_realm_id}, 1, {property}, %(time_end)s)\n        '
        ).format(default_realm_id=Literal(self.default_realm.id), property=Literal('stat3'))
        stat3: DependentCountStat = DependentCountStat('stat3', sql_data_collector(RealmCount, query, None), CountStat.HOUR, dependencies=['stat1', 'stat2'])
        query2: Callable[[Dict[str, Any]], SQL] = lambda kwargs: SQL(
            '\n            INSERT INTO analytics_realmcount (realm_id, value, property, end_time)\n            VALUES ({default_realm_id}, 1, {property}, %(time_end)s)\n        '
        ).format(default_realm_id=Literal(self.default_realm.id), property=Literal('stat4'))
        stat4: DependentCountStat = DependentCountStat('stat4', sql_data_collector(RealmCount, query2, None), CountStat.DAY, dependencies=['stat1', 'stat2'])
        dummy_count_stats: Dict[str, Any] = {'stat1': stat1, 'stat2': stat2, 'stat3': stat3, 'stat4': stat4}
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


class TestCountStats(AnalyticsTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.second_realm = do_create_realm(string_id='second-realm', name='Second Realm', date_created=self.TIME_ZERO - 2 * self.DAY)
        for minutes_ago in [0, 1, 61, 60 * 24 + 1]:
            creation_time: datetime = self.TIME_ZERO - minutes_ago * self.MINUTE
            user: UserProfile = self.create_user(email=f'user-{minutes_ago}@second.analytics', realm=self.second_realm, date_joined=creation_time)
            recipient: Recipient = self.create_stream_with_recipient(name=f'stream {minutes_ago}', realm=self.second_realm, date_created=creation_time)[1]
            self.create_message(user, recipient, date_sent=creation_time)
        self.hourly_user: UserProfile = get_user('user-1@second.analytics', self.second_realm)
        self.daily_user: UserProfile = get_user('user-61@second.analytics', self.second_realm)
        self.no_message_realm = do_create_realm(string_id='no-message-realm', name='No Message Realm', date_created=self.TIME_ZERO - 2 * self.DAY)
        self.create_user(realm=self.no_message_realm)
        self.create_stream_with_recipient(realm=self.no_message_realm)
        self.create_direct_message_group_with_recipient()

    def test_upload_quota_used_bytes(self) -> None:
        stat: Any = COUNT_STATS['upload_quota_used_bytes::day']
        self.current_property = stat.property
        user1: UserProfile = self.create_user()
        user2: UserProfile = self.create_user()
        user_second_realm: UserProfile = self.create_user(realm=self.second_realm)
        self.create_attachment(user1, 'file1', 100, self.TIME_LAST_HOUR, 'text/plain')
        attachment2: Attachment = self.create_attachment(user2, 'file2', 200, self.TIME_LAST_HOUR, 'text/plain')
        self.create_attachment(user_second_realm, 'file3', 10, self.TIME_LAST_HOUR, 'text/plain')
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['value', 'subgroup', 'realm'], [[300, None, self.default_realm], [10, None, self.second_realm]])
        attachment2.delete()
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO + self.DAY)
        self.assertTableState(
            RealmCount,
            ['value', 'subgroup', 'realm', 'end_time'],
            [[300, None, self.default_realm, self.TIME_ZERO], [10, None, self.second_realm, self.TIME_ZERO], [100, None, self.default_realm, self.TIME_ZERO + self.DAY], [10, None, self.second_realm, self.TIME_ZERO + self.DAY]],
        )

    def test_messages_sent_by_is_bot(self) -> None:
        stat: Any = COUNT_STATS['messages_sent:is_bot:hour']
        self.current_property = stat.property
        bot: UserProfile = self.create_user(is_bot=True)
        human1: UserProfile = self.create_user()
        human2: UserProfile = self.create_user()
        recipient_human1: Recipient = Recipient.objects.get(type_id=human1.id, type=Recipient.PERSONAL)
        recipient_stream: Recipient = self.create_stream_with_recipient()[1]
        recipient_direct_message_group: Recipient = self.create_direct_message_group_with_recipient()[1]
        self.create_message(bot, recipient_human1)
        self.create_message(bot, recipient_stream)
        self.create_message(bot, recipient_direct_message_group)
        self.create_message(human1, recipient_human1)
        self.create_message(human2, recipient_human1)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(UserCount, ['value', 'subgroup', 'user'], [[1, 'false', human1], [1, 'false', human2], [3, 'true', bot], [1, 'false', self.hourly_user]])
        self.assertTableState(RealmCount, ['value', 'subgroup', 'realm'], [[2, 'false'], [3, 'true'], [1, 'false', self.second_realm]])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [[3, 'false'], [3, 'true']])
        self.assertTableState(StreamCount, [], [])

    def test_messages_sent_by_is_bot_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['messages_sent:is_bot:hour']
        self.current_property = stat.property
        bot: UserProfile = self.create_user(is_bot=True)
        human1: UserProfile = self.create_user()
        human2: UserProfile = self.create_user()
        recipient_human1: Recipient = Recipient.objects.get(type_id=human1.id, type=Recipient.PERSONAL)
        recipient_stream: Recipient = self.create_stream_with_recipient()[1]
        recipient_direct_message_group: Recipient = self.create_direct_message_group_with_recipient()[1]
        self.create_message(bot, recipient_human1)
        self.create_message(bot, recipient_stream)
        self.create_message(bot, recipient_direct_message_group)
        self.create_message(human1, recipient_human1)
        self.create_message(human2, recipient_human1)
        self.create_message(self.hourly_user, recipient_human1)
        self.create_message(self.hourly_user, recipient_stream)
        self.create_message(self.hourly_user, recipient_direct_message_group)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        self.assertTableState(UserCount, ['value', 'subgroup', 'user'], [[1, 'false', human1], [1, 'false', human2], [3, 'true', bot]])
        self.assertTableState(RealmCount, ['value', 'subgroup', 'realm'], [[2, 'false', self.default_realm], [3, 'true', self.default_realm]])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [])
        self.assertTableState(StreamCount, [], [])

    def test_messages_sent_by_message_type(self) -> None:
        stat: Any = COUNT_STATS['messages_sent:message_type:day']
        self.current_property = stat.property
        user1: UserProfile = self.create_user(is_bot=True)
        user2: UserProfile = self.create_user()
        user3: UserProfile = self.create_user()
        recipient_stream1: Recipient = self.create_stream_with_recipient(invite_only=True)[1]
        recipient_stream2: Recipient = self.create_stream_with_recipient(invite_only=True)[1]
        self.create_message(user1, recipient_stream1)
        self.create_message(user2, recipient_stream1)
        self.create_message(user2, recipient_stream2)
        recipient_stream3: Recipient = self.create_stream_with_recipient()[1]
        recipient_stream4: Recipient = self.create_stream_with_recipient()[1]
        self.create_message(user1, recipient_stream3)
        self.create_message(user1, recipient_stream4)
        self.create_message(user2, recipient_stream3)
        recipient_direct_message_group1: Recipient = self.create_direct_message_group_with_recipient()[1]
        recipient_direct_message_group2: Recipient = self.create_direct_message_group_with_recipient()[1]
        self.create_message(user1, recipient_direct_message_group1)
        self.create_message(user2, recipient_direct_message_group2)
        recipient_user1: Recipient = Recipient.objects.get(type_id=user1.id, type=Recipient.PERSONAL)
        recipient_user2: Recipient = Recipient.objects.get(type_id=user2.id, type=Recipient.PERSONAL)
        recipient_user3: Recipient = Recipient.objects.get(type_id=user3.id, type=Recipient.PERSONAL)
        self.create_message(user1, recipient_user2)
        self.create_message(user2, recipient_user1)
        self.create_message(user3, recipient_user3)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(
            UserCount,
            ['value', 'subgroup', 'user'],
            [
                [1, 'private_stream', user1],
                [2, 'private_stream', user2],
                [2, 'public_stream', user1],
                [1, 'public_stream', user2],
                [1, 'private_message', user1],
                [1, 'private_message', user2],
                [1, 'private_message', user3],
                [1, 'huddle_message', user1],
                [1, 'huddle_message', user2],
                [1, 'public_stream', self.hourly_user],
                [1, 'public_stream', self.daily_user],
            ],
        )
        self.assertTableState(
            RealmCount,
            ['value', 'subgroup', 'realm'],
            [[3, 'private_stream'], [3, 'public_stream'], [3, 'private_message'], [2, 'huddle_message'], [2, 'public_stream', self.second_realm]],
        )
        self.assertTableState(
            InstallationCount,
            ['value', 'subgroup'],
            [[3, 'private_stream'], [5, 'public_stream'], [3, 'private_message'], [2, 'huddle_message']],
        )
        self.assertTableState(StreamCount, [], [])

    def test_messages_sent_by_message_type_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['messages_sent:message_type:day']
        self.current_property = stat.property
        user: UserProfile = self.create_user()
        user_recipient: Recipient = Recipient.objects.get(type_id=user.id, type=Recipient.PERSONAL)
        private_stream_recipient: Recipient = self.create_stream_with_recipient(invite_only=True)[1]
        stream_recipient: Recipient = self.create_stream_with_recipient()[1]
        direct_message_group_recipient: Recipient = self.create_direct_message_group_with_recipient()[1]
        self.create_message(user, user_recipient)
        self.create_message(user, private_stream_recipient)
        self.create_message(user, stream_recipient)
        self.create_message(user, direct_message_group_recipient)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        self.assertTableState(UserCount, ['value', 'subgroup', 'user'], [[1, 'private_message', user], [1, 'private_stream', user], [1, 'huddle_message', user], [1, 'public_stream', user]])
        self.assertTableState(RealmCount, ['value', 'subgroup'], [[1, 'private_message'], [1, 'private_stream'], [1, 'public_stream'], [1, 'huddle_message']])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [])
        self.assertTableState(StreamCount, [], [])

    def test_messages_sent_to_recipients_with_same_id(self) -> None:
        stat: Any = COUNT_STATS['messages_sent:message_type:day']
        self.current_property = stat.property
        user: UserProfile = self.create_user(id=1000)
        user_recipient: Recipient = Recipient.objects.get(type_id=user.id, type=Recipient.PERSONAL)
        stream_recipient: Recipient = self.create_stream_with_recipient(id=1000)[1]
        direct_message_group_recipient: Recipient = self.create_direct_message_group_with_recipient(id=1000)[1]
        self.create_message(user, user_recipient)
        self.create_message(user, stream_recipient)
        self.create_message(user, direct_message_group_recipient)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(
            UserCount,
            ['value', 'subgroup', 'user'],
            [[1, 'private_message', user], [1, 'huddle_message', user], [1, 'public_stream', user], [1, 'public_stream', self.hourly_user], [1, 'public_stream', self.daily_user]],
        )

    def test_messages_sent_by_client(self) -> None:
        stat: Any = COUNT_STATS['messages_sent:client:day']
        self.current_property = stat.property
        user1: UserProfile = self.create_user(is_bot=True)
        user2: UserProfile = self.create_user()
        recipient_user2: Recipient = Recipient.objects.get(type_id=user2.id, type=Recipient.PERSONAL)
        recipient_stream: Recipient = self.create_stream_with_recipient()[1]
        recipient_direct_message_group: Recipient = self.create_direct_message_group_with_recipient()[1]
        client2: Client = Client.objects.create(name='client2')
        self.create_message(user1, recipient_user2, sending_client=client2)
        self.create_message(user1, recipient_stream)
        self.create_message(user1, recipient_direct_message_group)
        self.create_message(user2, recipient_user2, sending_client=client2)
        self.create_message(user2, recipient_user2, sending_client=client2)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        client2_id: str = str(client2.id)
        website_client_id: str = str(get_client('website').id)
        self.assertTableState(
            UserCount,
            ['value', 'subgroup', 'user'],
            [[2, website_client_id, user1], [1, client2_id, user1], [2, client2_id, user2], [1, website_client_id, self.hourly_user], [1, website_client_id, self.daily_user]],
        )
        self.assertTableState(
            RealmCount,
            ['value', 'subgroup', 'realm'],
            [[2, website_client_id], [3, client2_id], [2, website_client_id, self.second_realm]],
        )
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [[4, website_client_id], [3, client2_id]])
        self.assertTableState(StreamCount, [], [])

    def test_messages_sent_by_client_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['messages_sent:client:day']
        self.current_property = stat.property
        user1: UserProfile = self.create_user(is_bot=True)
        user2: UserProfile = self.create_user()
        recipient_user2: Recipient = Recipient.objects.get(type_id=user2.id, type=Recipient.PERSONAL)
        client2: Client = Client.objects.create(name='client2')
        self.create_message(user1, recipient_user2, sending_client=client2)
        self.create_message(user2, recipient_user2, sending_client=client2)
        self.create_message(user2, recipient_user2)
        self.create_message(self.hourly_user, recipient_user2, sending_client=client2)
        self.create_message(self.hourly_user, recipient_user2, sending_client=client2)
        self.create_message(self.hourly_user, recipient_user2)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        client2_id: str = str(client2.id)
        website_client_id: str = str(get_client('website').id)
        self.assertTableState(UserCount, ['value', 'subgroup', 'user'], [[1, client2_id, user1], [1, client2_id, user2], [1, website_client_id, user2]])
        self.assertTableState(RealmCount, ['value', 'subgroup'], [[1, website_client_id], [2, client2_id]])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [])
        self.assertTableState(StreamCount, [], [])

    def test_messages_sent_to_stream_by_is_bot(self) -> None:
        stat: Any = COUNT_STATS['messages_in_stream:is_bot:day']
        self.current_property = stat.property
        bot: UserProfile = self.create_user(is_bot=True)
        human1: UserProfile = self.create_user()
        human2: UserProfile = self.create_user()
        recipient_human1: Recipient = Recipient.objects.get(type_id=human1.id, type=Recipient.PERSONAL)
        stream1, recipient_stream1 = self.create_stream_with_recipient()
        stream2, recipient_stream2 = self.create_stream_with_recipient()
        self.create_message(human1, recipient_stream1)
        self.create_message(human2, recipient_stream1)
        self.create_message(human1, recipient_stream2)
        self.create_message(bot, recipient_stream2)
        self.create_message(bot, recipient_stream2)
        self.create_message(human2, recipient_human1)
        self.create_message(bot, recipient_human1)
        recipient_direct_message_group: Recipient = self.create_direct_message_group_with_recipient()[1]
        self.create_message(human1, recipient_direct_message_group)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(StreamCount, ['value', 'subgroup', 'stream'], [[2, 'false', stream1], [1, 'false', stream2], [2, 'true', stream2], [1, 'false', Stream.objects.get(name='stream 1')], [1, 'false', Stream.objects.get(name='stream 61')]])
        self.assertTableState(RealmCount, ['value', 'subgroup', 'realm'], [[3, 'false'], [2, 'true'], [2, 'false', self.second_realm]])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [[5, 'false'], [2, 'true']])
        self.assertTableState(UserCount, [], [])

    def test_messages_sent_to_stream_by_is_bot_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['messages_in_stream:is_bot:day']
        self.current_property = stat.property
        human1: UserProfile = self.create_user()
        bot: UserProfile = self.create_user(is_bot=True)
        realm: Dict[str, Any] = {'realm': self.second_realm}
        stream1, recipient_stream1 = self.create_stream_with_recipient()
        stream2, recipient_stream2 = self.create_stream_with_recipient(**realm)
        self.create_message(human1, recipient_stream1)
        self.create_message(bot, recipient_stream1)
        self.create_message(self.hourly_user, recipient_stream2)
        self.create_message(self.daily_user, recipient_stream2)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        self.assertTableState(StreamCount, ['value', 'subgroup', 'stream'], [[1, 'false', stream1], [1, 'true', stream1]])
        self.assertTableState(RealmCount, ['value', 'subgroup', 'realm'], [[1, 'false'], [1, 'true']])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [])
        self.assertTableState(UserCount, [], [])

    def create_interval(self, user: UserProfile, start_offset: timedelta, end_offset: timedelta) -> None:
        UserActivityInterval.objects.create(user_profile=user, start=self.TIME_ZERO - start_offset, end=self.TIME_ZERO - end_offset)

    def test_1day_actives(self) -> None:
        stat: Any = COUNT_STATS['1day_actives::day']
        self.current_property = stat.property
        _1day: timedelta = 1 * self.DAY - UserActivityInterval.MIN_INTERVAL_LENGTH
        user1: UserProfile = self.create_user()
        self.create_interval(user1, _1day + self.DAY, _1day + timedelta(seconds=1))
        self.create_interval(user1, timedelta(0), -self.HOUR)
        user2: UserProfile = self.create_user()
        self.create_interval(user2, _1day + self.DAY, _1day)
        user3: UserProfile = self.create_user()
        self.create_interval(user3, 2 * self.DAY, 1 * self.DAY)
        self.create_interval(user3, 20 * self.HOUR, 19 * self.HOUR)
        self.create_interval(user3, 20 * self.MINUTE, 19 * self.MINUTE)
        user4: UserProfile = self.create_user()
        self.create_interval(user4, 1.5 * self.DAY, 0.5 * self.DAY)
        user5: UserProfile = self.create_user()
        self.create_interval(user5, self.MINUTE, -self.MINUTE)
        user6: UserProfile = self.create_user()
        self.create_interval(user6, 2 * self.DAY, -2 * self.DAY)
        user7: UserProfile = self.create_user(realm=self.second_realm)
        self.create_interval(user7, 20 * self.MINUTE, 19 * self.MINUTE)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(UserCount, ['value', 'user'], [[1, user2], [1, user3], [1, user4], [1, user5], [1, user6], [1, user7]])
        self.assertTableState(RealmCount, ['value', 'realm'], [[5, self.default_realm], [1, self.second_realm]])
        self.assertTableState(InstallationCount, ['value'], [[6]])
        self.assertTableState(StreamCount, [], [])

    def test_1day_actives_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['1day_actives::day']
        self.current_property = stat.property
        _1day: timedelta = 1 * self.DAY - UserActivityInterval.MIN_INTERVAL_LENGTH
        user1: UserProfile = self.create_user()
        user2: UserProfile = self.create_user()
        self.create_interval(user1, 20 * self.HOUR, 19 * self.HOUR)
        self.create_interval(user2, _1day + self.DAY, _1day)
        user3: UserProfile = self.create_user(realm=self.second_realm)
        self.create_interval(user3, 20 * self.MINUTE, 19 * self.MINUTE)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        self.assertTableState(UserCount, ['value', 'user'], [[1, user2], [1, user2]])
        self.assertTableState(RealmCount, ['value', 'realm'], [[2, self.default_realm]])
        self.assertTableState(InstallationCount, ['value'], [])
        self.assertTableState(StreamCount, [], [])

    def test_15day_actives(self) -> None:
        stat: Any = COUNT_STATS['15day_actives::day']
        self.current_property = stat.property
        _15day: timedelta = 15 * self.DAY - UserActivityInterval.MIN_INTERVAL_LENGTH
        user1: UserProfile = self.create_user()
        self.create_interval(user1, _15day + self.DAY, _15day + timedelta(seconds=1))
        self.create_interval(user1, timedelta(0), -self.HOUR)
        user2: UserProfile = self.create_user()
        self.create_interval(user2, _15day + self.DAY, _15day)
        user3: UserProfile = self.create_user()
        self.create_interval(user3, 20 * self.DAY, 19 * self.DAY)
        self.create_interval(user3, 20 * self.HOUR, 19 * self.HOUR)
        self.create_interval(user3, 20 * self.MINUTE, 19 * self.MINUTE)
        user4: UserProfile = self.create_user()
        self.create_interval(user4, 20 * self.DAY, 10 * self.DAY)
        user5: UserProfile = self.create_user()
        self.create_interval(user5, self.MINUTE, -self.MINUTE)
        user6: UserProfile = self.create_user()
        self.create_interval(user6, 20 * self.DAY, -2 * self.DAY)
        user7: UserProfile = self.create_user(realm=self.second_realm)
        self.create_interval(user7, 20 * self.MINUTE, 19 * self.MINUTE)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(UserCount, ['value', 'user'], [[1, user2], [1, user3], [1, user4], [1, user5], [1, user6], [1, user7]])
        self.assertTableState(RealmCount, ['value', 'realm'], [[5, self.default_realm], [1, self.second_realm]])
        self.assertTableState(InstallationCount, ['value'], [[6]])
        self.assertTableState(StreamCount, [], [])

    def test_15day_actives_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['15day_actives::day']
        self.current_property = stat.property
        _15day: timedelta = 15 * self.DAY - UserActivityInterval.MIN_INTERVAL_LENGTH
        user1: UserProfile = self.create_user()
        user2: UserProfile = self.create_user()
        user3: UserProfile = self.create_user(realm=self.second_realm)
        self.create_interval(user1, _15day + self.DAY, _15day)
        self.create_interval(user2, 20 * self.HOUR, 19 * self.HOUR)
        self.create_interval(user3, 20 * self.HOUR, 19 * self.HOUR)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        self.assertTableState(UserCount, ['value', 'user'], [[1, user1], [1, user2]])
        self.assertTableState(RealmCount, ['value', 'realm'], [[2, self.default_realm]])
        self.assertTableState(InstallationCount, ['value'], [])
        self.assertTableState(StreamCount, [], [])

    def test_minutes_active(self) -> None:
        stat: Any = COUNT_STATS['minutes_active::day']
        self.current_property = stat.property
        user1: UserProfile = self.create_user()
        self.create_interval(user1, 25 * self.HOUR, self.DAY)
        self.create_interval(user1, timedelta(0), -self.HOUR)
        user2: UserProfile = self.create_user()
        self.create_interval(user2, 20 * self.DAY, 19 * self.DAY)
        self.create_interval(user2, 20 * self.HOUR, 19 * self.HOUR)
        self.create_interval(user2, 20 * self.MINUTE, 19 * self.MINUTE)
        user3: UserProfile = self.create_user()
        self.create_interval(user3, 25 * self.HOUR, 22 * self.HOUR)
        self.create_interval(user3, self.MINUTE, -self.MINUTE)
        user4: UserProfile = self.create_user()
        self.create_interval(user4, 2 * self.DAY, -2 * self.DAY)
        user5: UserProfile = self.create_user()
        self.create_interval(user5, self.MINUTE, timedelta(seconds=30))
        self.create_interval(user5, timedelta(seconds=20), timedelta(seconds=10))
        user6: UserProfile = self.create_user(realm=self.second_realm)
        self.create_interval(user6, 20 * self.MINUTE, 19 * self.MINUTE)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO)
        self.assertTableState(UserCount, ['value', 'user'], [[61, user2], [121, user3], [24 * 60, user4], [1, user6]])
        self.assertTableState(RealmCount, ['value', 'realm'], [[61 + 121 + 24 * 60, self.default_realm], [1, self.second_realm]])
        self.assertTableState(InstallationCount, ['value'], [[61 + 121 + 24 * 60 + 1]])
        self.assertTableState(StreamCount, [], [])

    def test_minutes_active_realm_constraint(self) -> None:
        COUNT_STATS  # type: ignore
        stat: Any = get_count_stats(self.default_realm)['minutes_active::day']
        self.current_property = stat.property
        user1: UserProfile = self.create_user()
        user2: UserProfile = self.create_user()
        user3: UserProfile = self.create_user(realm=self.second_realm)
        self.create_interval(user1, 20 * self.HOUR, 19 * self.HOUR)
        self.create_interval(user2, 20 * self.MINUTE, 19 * self.MINUTE)
        self.create_interval(user3, 20 * self.MINUTE, 19 * self.MINUTE)
        do_fill_count_stat_at_hour(stat, self.TIME_ZERO, self.default_realm)
        self.assertTableState(UserCount, ['value', 'user'], [[60, user1], [1, user2]])
        self.assertTableState(RealmCount, ['value', 'realm'], [[60 + 1, self.default_realm]])
        self.assertTableState(InstallationCount, ['value'], [])
        self.assertTableState(StreamCount, [], [])

    def test_last_successful_fill(self) -> None:
        self.assertIsNone(COUNT_STATS['messages_sent:is_bot:hour'].last_successful_fill())
        a_time: datetime = datetime(2016, 3, 14, 19, tzinfo=timezone.utc)
        one_hour_before: datetime = datetime(2016, 3, 14, 18, tzinfo=timezone.utc)
        one_day_before: datetime = datetime(2016, 3, 13, 19, tzinfo=timezone.utc)
        fillstate = FillState.objects.create(property=COUNT_STATS['messages_sent:is_bot:hour'].property, end_time=a_time, state=FillState.DONE)
        self.assertEqual(COUNT_STATS['messages_sent:is_bot:hour'].last_successful_fill(), a_time)
        fillstate.state = FillState.STARTED
        fillstate.save(update_fields=['state'])
        self.assertEqual(COUNT_STATS['messages_sent:is_bot:hour'].last_successful_fill(), one_hour_before)
        fillstate.property = COUNT_STATS['7day_actives::day'].property
        fillstate.save(update_fields=['property'])
        self.assertEqual(COUNT_STATS['7day_actives::day'].last_successful_fill(), one_day_before)


class TestDoAggregateToSummaryTable(AnalyticsTestCase):
    def test_no_aggregated_zeros(self) -> None:
        stat: LoggingCountStat = LoggingCountStat('test stat', UserCount, CountStat.HOUR)
        do_aggregate_to_summary_table(stat, self.TIME_ZERO)
        self.assertFalse(RealmCount.objects.exists())
        self.assertFalse(InstallationCount.objects.exists())


class TestDoIncrementLoggingStat(AnalyticsTestCase):
    def test_table_and_id_args(self) -> None:
        self.current_property = 'test'
        second_realm = do_create_realm(string_id='moo', name='moo')
        stat: LoggingCountStat = LoggingCountStat('test', RealmCount, CountStat.DAY)
        do_increment_logging_stat(self.default_realm, stat, None, self.TIME_ZERO)
        do_increment_logging_stat(second_realm, stat, None, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['realm'], [[self.default_realm], [second_realm]])
        user1: UserProfile = self.create_user()
        user2: UserProfile = self.create_user()
        stat = LoggingCountStat('test', UserCount, CountStat.DAY)
        do_increment_logging_stat(user1, stat, None, self.TIME_ZERO)
        do_increment_logging_stat(user2, stat, None, self.TIME_ZERO)
        self.assertTableState(UserCount, ['user'], [[user1], [user2]])
        stream1: Stream = self.create_stream_with_recipient()[0]
        stream2: Stream = self.create_stream_with_recipient()[0]
        stat = LoggingCountStat('test', StreamCount, CountStat.DAY)
        do_increment_logging_stat(stream1, stat, None, self.TIME_ZERO)
        do_increment_logging_stat(stream2, stat, None, self.TIME_ZERO)
        self.assertTableState(StreamCount, ['stream'], [[stream1], [stream2]])

    def test_frequency(self) -> None:
        times: List[datetime] = [self.TIME_ZERO - self.MINUTE * i for i in [0, 1, 61, 24 * 60 + 1]]
        stat: LoggingCountStat = LoggingCountStat('day test', RealmCount, CountStat.DAY)
        for time_ in times:
            do_increment_logging_stat(self.default_realm, stat, None, time_)
        stat = LoggingCountStat('hour test', RealmCount, CountStat.HOUR)
        for time_ in times:
            do_increment_logging_stat(self.default_realm, stat, None, time_)
        self.assertTableState(RealmCount, ['value', 'property', 'end_time'], [[3, 'day test', self.TIME_ZERO], [1, 'day test', self.TIME_ZERO - self.DAY], [2, 'hour test', self.TIME_ZERO], [1, 'hour test', self.TIME_LAST_HOUR], [1, 'hour test', self.TIME_ZERO - self.DAY]])

    def test_get_or_create(self) -> None:
        stat: LoggingCountStat = LoggingCountStat('test', RealmCount, CountStat.HOUR)
        do_increment_logging_stat(self.default_realm, stat, 'subgroup1', self.TIME_ZERO)
        do_increment_logging_stat(self.default_realm, stat, 'subgroup2', self.TIME_ZERO)
        do_increment_logging_stat(self.default_realm, stat, 'subgroup1', self.TIME_LAST_HOUR)
        self.current_property = 'test'
        self.assertTableState(RealmCount, ['value', 'subgroup', 'end_time'], [[1, 'subgroup1', self.TIME_ZERO], [1, 'subgroup2', self.TIME_ZERO], [1, 'subgroup1', self.TIME_LAST_HOUR]])
        do_increment_logging_stat(self.default_realm, stat, 'subgroup1', self.TIME_ZERO)
        self.assertTableState(RealmCount, ['value', 'subgroup', 'end_time'], [[2, 'subgroup1', self.TIME_ZERO], [1, 'subgroup2', self.TIME_ZERO], [1, 'subgroup1', self.TIME_LAST_HOUR]])

    def test_increment(self) -> None:
        stat: LoggingCountStat = LoggingCountStat('test', RealmCount, CountStat.DAY)
        self.current_property = 'test'
        do_increment_logging_stat(self.default_realm, stat, None, self.TIME_ZERO, increment=-1)
        self.assertTableState(RealmCount, ['value'], [[-1]])
        do_increment_logging_stat(self.default_realm, stat, None, self.TIME_ZERO, increment=3)
        self.assertTableState(RealmCount, ['value'], [[2]])
        do_increment_logging_stat(self.default_realm, stat, None, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['value'], [[3]])

    def test_do_increment_logging_start_query_count(self) -> None:
        stat: LoggingCountStat = LoggingCountStat('test', RealmCount, CountStat.DAY)
        with self.assert_database_query_count(1):
            do_increment_logging_stat(self.default_realm, stat, None, self.TIME_ZERO)


class TestLoggingCountStats(AnalyticsTestCase):
    def test_aggregation(self) -> None:
        stat: LoggingCountStat = LoggingCountStat('realm test', RealmCount, CountStat.DAY)
        do_increment_logging_stat(self.default_realm, stat, None, self.TIME_ZERO)
        process_count_stat(stat, self.TIME_ZERO)
        user: UserProfile = self.create_user()
        stat = LoggingCountStat('user test', UserCount, CountStat.DAY)
        do_increment_logging_stat(user, stat, None, self.TIME_ZERO)
        process_count_stat(stat, self.TIME_ZERO)
        stream: Stream = self.create_stream_with_recipient()[0]
        stat = LoggingCountStat('stream test', StreamCount, CountStat.DAY)
        do_increment_logging_stat(stream, stat, None, self.TIME_ZERO)
        process_count_stat(stat, self.TIME_ZERO)
        self.assertTableState(InstallationCount, ['property', 'value'], [['realm test', 1], ['user test', 1], ['stream test', 1]])
        self.assertTableState(RealmCount, ['property', 'value'], [['realm test', 1], ['user test', 1], ['stream test', 1]])
        self.assertTableState(UserCount, ['property', 'value'], [['user test', 1]])
        self.assertTableState(StreamCount, ['property', 'value'], [['stream test', 1]])

    @activate_push_notification_service()
    def test_mobile_pushes_received_count(self) -> None:
        self.server_uuid: str = '6cde5f7a-1f7e-4978-9716-49f69ebfc9fe'
        self.server: RemoteZulipServer = RemoteZulipServer.objects.create(uuid=self.server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', last_updated=timezone_now())
        hamlet: UserProfile = self.example_user('hamlet')
        token: str = 'aaaa'
        RemotePushDeviceToken.objects.create(kind=RemotePushDeviceToken.FCM, token=hex_to_b64(token), user_uuid=hamlet.uuid, server=self.server)
        RemotePushDeviceToken.objects.create(kind=RemotePushDeviceToken.FCM, token=hex_to_b64(token + 'aa'), user_uuid=hamlet.uuid, server=self.server)
        RemotePushDeviceToken.objects.create(kind=RemotePushDeviceToken.APNS, token=hex_to_b64(token), user_uuid=str(hamlet.uuid), server=self.server)
        message: Message = Message(
            sender=hamlet,
            recipient=self.example_user('othello').recipient,
            realm_id=hamlet.realm_id,
            content='This is test content',
            rendered_content='This is test content',
            date_sent=timezone_now(),
            sending_client=get_client('test'),
        )
        message.set_topic_name('Test topic')
        message.save()
        gcm_payload, gcm_options = get_message_payload_gcm(hamlet, message)
        apns_payload = get_message_payload_apns(hamlet, message, NotificationTriggers.DIRECT_MESSAGE)
        payload: Dict[str, Any] = {'user_id': hamlet.id, 'user_uuid': str(hamlet.uuid), 'gcm_payload': gcm_payload, 'apns_payload': apns_payload, 'gcm_options': gcm_options}
        now: datetime = timezone_now()
        with time_machine.travel(now, tick=False), \
             mock.patch('zilencer.views.send_android_push_notification', return_value=1), \
             mock.patch('zilencer.views.send_apple_push_notification', return_value=1), \
             mock.patch('corporate.lib.stripe.RemoteServerBillingSession.current_count_for_billed_licenses', return_value=10), \
             self.assertLogs('zilencer.views', level='INFO'):
            result = self.uuid_post(self.server_uuid, '/api/v1/remotes/push/notify', payload, content_type='application/json', subdomain='')
            self.assert_json_success(result)
        self.assertTableState(RemoteInstallationCount, ['property', 'value', 'subgroup', 'server', 'remote_id', 'end_time'], [['mobile_pushes_received::day', 3, None, self.server, None, ceiling_to_day(now)], ['mobile_pushes_forwarded::day', 2, None, self.server, None, ceiling_to_day(now)]])
        self.assertFalse(RemoteRealmCount.objects.filter(property='mobile_pushes_received::day').exists())
        self.assertFalse(RemoteRealmCount.objects.filter(property='mobile_pushes_forwarded::day').exists())
        payload = {'user_id': hamlet.id, 'user_uuid': str(hamlet.uuid), 'realm_uuid': str(hamlet.realm.uuid), 'gcm_payload': gcm_payload, 'apns_payload': apns_payload, 'gcm_options': gcm_options}
        with time_machine.travel(now, tick=False), \
             mock.patch('zilencer.views.send_android_push_notification', return_value=1), \
             mock.patch('zilencer.views.send_apple_push_notification', return_value=1), \
             mock.patch('corporate.lib.stripe.RemoteServerBillingSession.current_count_for_billed_licenses', return_value=10), \
             self.assertLogs('zilencer.views', level='INFO'):
            result = self.uuid_post(self.server_uuid, '/api/v1/remotes/push/notify', payload, content_type='application/json', subdomain='')
            self.assert_json_success(result)
        self.assertTableState(RemoteInstallationCount, ['property', 'value', 'subgroup', 'server', 'remote_id', 'end_time'], [['mobile_pushes_received::day', 6, None, self.server, None, ceiling_to_day(now)], ['mobile_pushes_forwarded::day', 4, None, self.server, None, ceiling_to_day(now)]])
        self.assertFalse(RemoteRealmCount.objects.filter(property='mobile_pushes_received::day').exists())
        self.assertFalse(RemoteRealmCount.objects.filter(property='mobile_pushes_forwarded::day').exists())
        realm = hamlet.realm
        remote_realm: RemoteRealm = RemoteRealm.objects.create(server=self.server, uuid=realm.uuid, uuid_owner_secret=realm.uuid_owner_secret, host=realm.host, realm_deactivated=realm.deactivated, realm_date_created=realm.date_created)
        with time_machine.travel(now, tick=False), \
             mock.patch('zilencer.views.send_android_push_notification', return_value=1), \
             mock.patch('zilencer.views.send_apple_push_notification', return_value=1), \
             mock.patch('corporate.lib.stripe.RemoteRealmBillingSession.current_count_for_billed_licenses', return_value=10), \
             self.assertLogs('zilencer.views', level='INFO'):
            result = self.uuid_post(self.server_uuid, '/api/v1/remotes/push/notify', payload, content_type='application/json', subdomain='')
            self.assert_json_success(result)
        self.assertTableState(RemoteInstallationCount, ['property', 'value', 'subgroup', 'server', 'remote_id', 'end_time'], [['mobile_pushes_received::day', 9, None, self.server, None, ceiling_to_day(now)], ['mobile_pushes_forwarded::day', 6, None, self.server, None, ceiling_to_day(now)]])
        self.assertTableState(RemoteRealmCount, ['property', 'value', 'subgroup', 'server', 'remote_realm', 'remote_id', 'end_time'], [['mobile_pushes_received::day', 3, None, self.server, remote_realm, None, ceiling_to_day(now)], ['mobile_pushes_forwarded::day', 2, None, self.server, remote_realm, None, ceiling_to_day(now)]])

    def test_invites_sent(self) -> None:
        property: str = 'invites_sent::day'

        @contextmanager
        def invite_context(too_many_recent_realm_invites: bool = False, failure: bool = False) -> Iterator[None]:
            managers: List[AbstractContextManager[Any]] = [mock.patch('zerver.actions.invites.too_many_recent_realm_invites', return_value=False), self.captureOnCommitCallbacks(execute=True)]
            if failure:
                managers.append(self.assertRaises(InvitationError))
            with ExitStack() as stack:
                for mgr in managers:
                    stack.enter_context(mgr)
                yield

        def assertInviteCountEquals(count: int) -> None:
            self.assertEqual(count, RealmCount.objects.filter(property=property, subgroup=None).aggregate(Sum('value'))['value__sum'])
        user: UserProfile = self.create_user(email='first@domain.tld')
        stream, _ = self.create_stream_with_recipient()
        invite_expires_in_minutes: int = 2 * 24 * 60
        with invite_context():
            do_invite_users(user, ['user1@domain.tld', 'user2@domain.tld'], [stream], include_realm_default_subscriptions=False, invite_expires_in_minutes=invite_expires_in_minutes)
        assertInviteCountEquals(2)
        with invite_context():
            do_invite_users(user, ['user1@domain.tld', 'user2@domain.tld'], [stream], include_realm_default_subscriptions=False, invite_expires_in_minutes=invite_expires_in_minutes)
        assertInviteCountEquals(4)
        with invite_context(failure=True):
            do_invite_users(user, ['user3@domain.tld', 'malformed'], [stream], include_realm_default_subscriptions=False, invite_expires_in_minutes=invite_expires_in_minutes)
        assertInviteCountEquals(4)
        with invite_context():
            skipped = do_invite_users(user, ['first@domain.tld', 'user4@domain.tld'], [stream], include_realm_default_subscriptions=False, invite_expires_in_minutes=invite_expires_in_minutes)
            self.assert_length(skipped, 1)
        assertInviteCountEquals(5)
        do_revoke_user_invite(assert_is_not_none(PreregistrationUser.objects.filter(realm=user.realm).first()))
        assertInviteCountEquals(5)
        with invite_context():
            do_send_user_invite_email(assert_is_not_none(PreregistrationUser.objects.first()))
        assertInviteCountEquals(6)

    def test_messages_read_hour(self) -> None:
        read_count_property: str = 'messages_read::hour'
        interactions_property: str = 'messages_read_interactions::hour'
        user1: UserProfile = self.create_user()
        user2: UserProfile = self.create_user()
        stream, _ = self.create_stream_with_recipient()
        self.subscribe(user1, stream.name)
        self.subscribe(user2, stream.name)
        self.send_personal_message(user1, user2)
        do_mark_all_as_read(user2)
        self.assertEqual(1, UserCount.objects.filter(property=read_count_property).aggregate(Sum('value'))['value__sum'])
        self.assertEqual(1, UserCount.objects.filter(property=interactions_property).aggregate(Sum('value'))['value__sum'])
        self.send_stream_message(user1, stream.name)
        self.send_stream_message(user1, stream.name)
        do_mark_stream_messages_as_read(user2, assert_is_not_none(stream.recipient_id))
        self.assertEqual(3, UserCount.objects.filter(property=read_count_property).aggregate(Sum('value'))['value__sum'])
        self.assertEqual(2, UserCount.objects.filter(property=interactions_property).aggregate(Sum('value'))['value__sum'])
        message: Message = self.send_stream_message(user2, stream.name)
        do_update_message_flags(user1, 'add', 'read', [message])
        self.assertEqual(4, UserCount.objects.filter(property=read_count_property).aggregate(Sum('value'))['value__sum'])
        self.assertEqual(3, UserCount.objects.filter(property=interactions_property).aggregate(Sum('value'))['value__sum'])


class TestDeleteStats(AnalyticsTestCase):
    def test_do_drop_all_analytics_tables(self) -> None:
        user: UserProfile = self.create_user()
        stream: Stream = self.create_stream_with_recipient()[0]
        count_args: Dict[str, Any] = {'property': 'test', 'end_time': self.TIME_ZERO, 'value': 10}
        UserCount.objects.create(user=user, realm=user.realm, **count_args)
        StreamCount.objects.create(stream=stream, realm=stream.realm, **count_args)
        RealmCount.objects.create(realm=user.realm, **count_args)
        InstallationCount.objects.create(**count_args)
        FillState.objects.create(property='test', end_time=self.TIME_ZERO, state=FillState.DONE)
        analytics = apps.get_app_config('analytics')
        for table in analytics.models.values():
            self.assertTrue(table._default_manager.exists())
        do_drop_all_analytics_tables()
        for table in analytics.models.values():
            self.assertFalse(table._default_manager.exists())

    def test_do_drop_single_stat(self) -> None:
        user: UserProfile = self.create_user()
        stream: Stream = self.create_stream_with_recipient()[0]
        count_args_to_delete: Dict[str, Any] = {'property': 'to_delete', 'end_time': self.TIME_ZERO, 'value': 10}
        count_args_to_save: Dict[str, Any] = {'property': 'to_save', 'end_time': self.TIME_ZERO, 'value': 10}
        for count_args in [count_args_to_delete, count_args_to_save]:
            UserCount.objects.create(user=user, realm=user.realm, **count_args)
            StreamCount.objects.create(stream=stream, realm=stream.realm, **count_args)
            RealmCount.objects.create(realm=user.realm, **count_args)
            InstallationCount.objects.create(**count_args)
        FillState.objects.create(property='to_delete', end_time=self.TIME_ZERO, state=FillState.DONE)
        FillState.objects.create(property='to_save', end_time=self.TIME_ZERO, state=FillState.DONE)
        analytics = apps.get_app_config('analytics')
        for table in analytics.models.values():
            self.assertTrue(table._default_manager.exists())
        do_drop_single_stat('to_delete')
        for table in analytics.models.values():
            self.assertFalse(table._default_manager.filter(property='to_delete').exists())
            self.assertTrue(table._default_manager.filter(property='to_save').exists())


class TestActiveUsersAudit(AnalyticsTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.user: UserProfile = self.create_user(skip_auditlog=True)
        self.stat: Any = COUNT_STATS['active_users_audit:is_bot:day']
        self.current_property = self.stat.property

    def add_event(self, event_type: str, days_offset: float, user: Optional[UserProfile] = None) -> None:
        hours_offset: int = int(24 * days_offset)
        if user is None:
            user = self.user
        RealmAuditLog.objects.create(realm=user.realm, modified_user=user, event_type=event_type, event_time=self.TIME_ZERO - hours_offset * self.HOUR)

    def test_user_deactivated_in_future(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 1)
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 0)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup'], [['false']])

    def test_user_reactivated_in_future(self) -> None:
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 1)
        self.add_event(AuditLogEventType.USER_REACTIVATED, 0)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, [], [])

    def test_user_active_then_deactivated_same_day(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 1)
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 0.5)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, [], [])

    def test_user_inactive_then_activated_same_day(self) -> None:
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 1)
        self.add_event(AuditLogEventType.USER_REACTIVATED, 0.5)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup'], [['false']])

    def test_user_active_then_deactivated_with_day_gap(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 2)
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 1)
        process_count_stat(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup', 'end_time'], [['false', self.TIME_ZERO - self.DAY]])

    def test_user_deactivated_then_reactivated_with_day_gap(self) -> None:
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 2)
        self.add_event(AuditLogEventType.USER_REACTIVATED, 1)
        process_count_stat(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup'], [['false']])

    def test_event_types(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 4)
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 3)
        self.add_event(AuditLogEventType.USER_ACTIVATED, 2)
        self.add_event(AuditLogEventType.USER_REACTIVATED, 1)
        for i in range(4):
            do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO - i * self.DAY)
        self.assertTableState(RealmCount, ['subgroup', 'end_time'], [['false', self.TIME_ZERO - i * self.DAY] for i in [3, 1, 0]])

    def test_multiple_users_realms_and_bots(self) -> None:
        user1: UserProfile = self.create_user(skip_auditlog=True)
        user2: UserProfile = self.create_user(skip_auditlog=True)
        second_realm = do_create_realm(string_id='moo', name='moo')
        user3: UserProfile = self.create_user(skip_auditlog=True, realm=second_realm)
        user4: UserProfile = self.create_user(skip_auditlog=True, realm=second_realm, is_bot=True)
        for user in [user1, user2, user3, user4]:
            self.add_event(AuditLogEventType.USER_CREATED, 1, user=user)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['value', 'subgroup', 'realm'], [[2, 'false', self.default_realm], [1, 'false', second_realm], [1, 'true', second_realm]])
        self.assertTableState(InstallationCount, ['value', 'subgroup'], [[3, 'false'], [1, 'true']])
        self.assertTableState(StreamCount, [], [])

    def test_update_from_two_days_ago(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 2)
        process_count_stat(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup', 'end_time'], [['false', self.TIME_ZERO], ['false', self.TIME_ZERO - self.DAY]])

    def test_empty_realm_or_user_with_no_relevant_activity(self) -> None:
        self.add_event(AuditLogEventType.USER_SOFT_ACTIVATED, 1)
        self.create_user(skip_auditlog=True)
        do_create_realm(string_id='moo', name='moo')
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, [], [])

    def test_max_audit_entry_is_unrelated(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 1)
        self.add_event(AuditLogEventType.USER_SOFT_ACTIVATED, 0.5)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup'], [['false']])

    def test_simultaneous_unrelated_audit_entry(self) -> None:
        self.add_event(AuditLogEventType.USER_CREATED, 1)
        self.add_event(AuditLogEventType.USER_SOFT_ACTIVATED, 1)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['subgroup'], [['false']])

    def test_simultaneous_max_audit_entries_of_different_users(self) -> None:
        user1: UserProfile = self.create_user(skip_auditlog=True)
        user2: UserProfile = self.create_user(skip_auditlog=True)
        user3: UserProfile = self.create_user(skip_auditlog=True)
        self.add_event(AuditLogEventType.USER_CREATED, 0.5, user=user1)
        self.add_event(AuditLogEventType.USER_CREATED, 0.5, user=user2)
        self.add_event(AuditLogEventType.USER_CREATED, 1, user=user3)
        self.add_event(AuditLogEventType.USER_DEACTIVATED, 0.5, user=user3)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, ['value', 'subgroup'], [[2, 'false']])

    def test_end_to_end_with_actions_dot_py(self) -> None:
        do_create_user('email1', 'password', self.default_realm, 'full_name', acting_user=None)
        user2: UserProfile = do_create_user('email2', 'password', self.default_realm, 'full_name', acting_user=None)
        user3: UserProfile = do_create_user('email3', 'password', self.default_realm, 'full_name', acting_user=None)
        do_deactivate_user(user3, acting_user=None)
        user3.is_mirror_dummy = True
        user3.save(update_fields=['is_mirror_dummy'])
        user4: UserProfile = do_create_user('email4', 'password', self.default_realm, 'full_name', acting_user=None)
        do_deactivate_user(user2, acting_user=None)
        do_activate_mirror_dummy_user(user3, acting_user=None)
        do_reactivate_user(user4, acting_user=None)
        end_time: datetime = floor_to_day(timezone_now()) + self.DAY
        do_fill_count_stat_at_hour(self.stat, end_time)
        self.assertTrue(RealmCount.objects.filter(realm=self.default_realm, property=self.current_property, subgroup='false', end_time=end_time, value=3).exists())


class TestRealmActiveHumans(AnalyticsTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.stat: Any = COUNT_STATS['realm_active_humans::day']
        self.current_property = self.stat.property

    def mark_15day_active(self, user: UserProfile, end_time: Optional[datetime] = None) -> None:
        if end_time is None:
            end_time = self.TIME_ZERO
        UserCount.objects.create(user=user, realm=user.realm, property='15day_actives::day', end_time=end_time, value=1)

    def test_basic_logic(self) -> None:
        user: UserProfile = self.create_user()
        self.mark_15day_active(user, end_time=self.TIME_ZERO)
        self.mark_15day_active(user, end_time=self.TIME_ZERO + self.DAY)
        for i in [-1, 0, 1]:
            do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO + i * self.DAY)
        self.assertTableState(RealmCount, ['value', 'end_time'], [[1, self.TIME_ZERO], [1, self.TIME_ZERO + self.DAY]])

    def test_bots_not_counted(self) -> None:
        bot: UserProfile = self.create_user(is_bot=True)
        self.mark_15day_active(bot)
        do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO)
        self.assertTableState(RealmCount, [], [])

    def test_multiple_users_realms_and_times(self) -> None:
        user1: UserProfile = self.create_user(date_joined=self.TIME_ZERO - 2 * self.DAY)
        user2: UserProfile = self.create_user(date_joined=self.TIME_ZERO - 2 * self.DAY)
        second_realm = do_create_realm(string_id='second', name='second')
        user3: UserProfile = self.create_user(date_joined=self.TIME_ZERO - 2 * self.DAY, realm=second_realm)
        user4: UserProfile = self.create_user(date_joined=self.TIME_ZERO - 2 * self.DAY, realm=second_realm)
        user5: UserProfile = self.create_user(date_joined=self.TIME_ZERO - 2 * self.DAY, realm=second_realm)
        for user in [user1, user3, user4]:
            self.mark_15day_active(user, end_time=self.TIME_ZERO - self.DAY)
        for user in [user1, user2, user3, user4, user5]:
            self.mark_15day_active(user)
        for i in [-1, 0, 1]:
            do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO + i * self.DAY)
        self.assertTableState(RealmCount, ['value', 'realm', 'end_time'], [[1, self.default_realm, self.TIME_ZERO - self.DAY], [2, second_realm, self.TIME_ZERO - self.DAY], [2, self.default_realm, self.TIME_ZERO], [3, second_realm, self.TIME_ZERO]])
        self.create_user()
        third_realm = do_create_realm(string_id='third', name='third')
        self.create_user(realm=third_realm)
        RealmCount.objects.all().delete()
        InstallationCount.objects.all().delete()
        for i in [-1, 0, 1]:
            do_fill_count_stat_at_hour(self.stat, self.TIME_ZERO + i * self.DAY)
        self.assertTableState(RealmCount, ['value', 'realm', 'end_time'], [[1, self.default_realm, self.TIME_ZERO - self.DAY], [2, second_realm, self.TIME_ZERO - self.DAY], [2, self.default_realm, self.TIME_ZERO], [3, second_realm, self.TIME_ZERO]])

    def test_end_to_end(self) -> None:
        user1: UserProfile = do_create_user('email1', 'password', self.default_realm, 'full_name', acting_user=None)
        user2: UserProfile = do_create_user('email2', 'password', self.default_realm, 'full_name', acting_user=None)
        do_create_user('email3', 'password', self.default_realm, 'full_name', acting_user=None)
        time_zero: datetime = floor_to_day(timezone_now()) + self.DAY
        update_user_activity_interval(user1, time_zero)
        update_user_activity_interval(user2, time_zero)
        do_deactivate_user(user2, acting_user=None)
        for property in ['active_users_audit:is_bot:day', '15day_actives::day', 'realm_active_humans::day']:
            FillState.objects.create(property=property, state=FillState.DONE, end_time=time_zero)
            process_count_stat(COUNT_STATS[property], time_zero + self.DAY)
        self.assertEqual(RealmCount.objects.filter(property='realm_active_humans::day', end_time=time_zero + self.DAY, value=1).count(), 1)
        self.assertEqual(RealmCount.objects.filter(property='realm_active_humans::day').count(), 1)


class GetLastIdFromServerTest(ZulipTestCase):
    def test_get_last_id_from_server_ignores_null(self) -> None:
        """
        Verifies that get_last_id_from_server ignores null remote_ids, since this goes
        against the default Postgres ordering behavior, which treats nulls as the largest value.
        """
        self.server_uuid: str = '6cde5f7a-1f7e-4978-9716-49f69ebfc9fe'
        self.server: RemoteZulipServer = RemoteZulipServer.objects.create(uuid=self.server_uuid, api_key='magic_secret_api_key', hostname='demo.example.com', last_updated=timezone_now())
        first = RemoteInstallationCount.objects.create(end_time=timezone_now(), server=self.server, property='test', value=1, remote_id=1)
        RemoteInstallationCount.objects.create(end_time=timezone_now(), server=self.server, property='test2', value=1, remote_id=None)
        result: Optional[int] = get_last_id_from_server(self.server, RemoteInstallationCount)
        self.assertEqual(result, first.remote_id)