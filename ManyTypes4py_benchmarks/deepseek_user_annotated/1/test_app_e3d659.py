import socket
import sys
from typing import Any, Mapping, NamedTuple, Optional, Type, Dict, List, Union, cast
from pathlib import Path

import faust
import mode
import pytest
import pytz

from mode.supervisors import OneForAllSupervisor
from mode.utils.mocks import patch
from yarl import URL

from faust import App
from faust.app import BootStrategy
from faust.assignor import LeaderAssignor, PartitionAssignor
from faust.exceptions import AlreadyConfiguredWarning, ImproperlyConfigured
from faust.app.router import Router
from faust.sensors import Monitor
from faust.serializers import Registry
from faust.tables import TableManager
from faust.transport.utils import DefaultSchedulingStrategy
from faust.types.enums import ProcessingGuarantee
from faust.types.settings import Settings
from faust.types.settings.params import Param
from faust.types.web import ResourceOptions

TABLEDIR: Path
DATADIR: Path
if sys.platform == 'win32':
    TABLEDIR = Path('c:/Program Files/Faust/')
    DATADIR = Path('c:/Temporary Files/Faust/')
else:
    DATADIR = Path('/etc/faust/')
    TABLEDIR = Path('/var/faust/')


class OtherSchedulingStrategy(DefaultSchedulingStrategy):
    ...


def _dummy_partitioner(a: Any, b: Any, c: Any) -> int:
    return 0


class EnvCase(NamedTuple):
    env: Mapping[str, str]
    setting: Param[Any]
    expected_value: Any


class test_settings:

    def App(self, id: str = 'myid', **kwargs: Any) -> App:
        app = App(id, **kwargs)
        app.finalize()
        return app

    def test_env_with_prefix(self) -> None:
        env = {
            'FOO_BROKER_URL': 'foobar://',
        }
        app = self.App(env=env, env_prefix='FOO_', broker='xaz://')
        assert app.conf.broker == [URL('foobar://')]

    @pytest.mark.parametrize('env,setting,expected_value', [
        EnvCase(
            env={'APP_DATADIR': '/foo/bar/baz'},
            setting=Settings.datadir,
            expected_value=Path('/foo/bar/baz'),
        ),
        EnvCase(
            env={'APP_TABLEDIR': '/foo/bar/bax'},
            setting=Settings.tabledir,
            expected_value=Path('/foo/bar/bax'),
        ),
        EnvCase(
            env={'APP_DEBUG': 'yes'},
            setting=Settings.debug,
            expected_value=True,
        ),
        EnvCase(
            env={'APP_DEBUG': 'no'},
            setting=Settings.debug,
            expected_value=False,
        ),
        EnvCase(
            env={'APP_DEBUG': '0'},
            setting=Settings.debug,
            expected_value=False,
        ),
        EnvCase(
            env={'APP_DEBUG': ''},
            setting=Settings.debug,
            expected_value=False,
        ),
        EnvCase(
            env={'TIMEZONE': 'Europe/Berlin'},
            setting=Settings.timezone,
            expected_value=pytz.timezone('Europe/Berlin'),
        ),
        EnvCase(
            env={'APP_VERSION': '3'},
            setting=Settings.version,
            expected_value=3,
        ),
        EnvCase(
            env={'AGENT_SUPERVISOR': 'mode.supervisors.OneForAllSupervisor'},
            setting=Settings.agent_supervisor,
            expected_value=OneForAllSupervisor,
        ),
        EnvCase(
            env={'BLOCKING_TIMEOUT': '0.0'},
            setting=Settings.blocking_timeout,
            expected_value=0.0,
        ),
        EnvCase(
            env={'BLOCKING_TIMEOUT': '3.03'},
            setting=Settings.blocking_timeout,
            expected_value=3.03,
        ),
        EnvCase(
            env={'BROKER_URL': 'foo://'},
            setting=Settings.broker,
            expected_value=[URL('foo://')],
        ),
        EnvCase(
            env={'BROKER_URL': 'foo://a;foo://b;foo://c'},
            setting=Settings.broker,
            expected_value=[URL('foo://a'), URL('foo://b'), URL('foo://c')],
        ),
        EnvCase(
            env={'BROKER_URL': 'foo://a;foo://b;foo://c'},
            setting=Settings.broker_consumer,
            expected_value=[URL('foo://a'), URL('foo://b'), URL('foo://c')],
        ),
        EnvCase(
            env={'BROKER_URL': 'foo://a;foo://b;foo://c'},
            setting=Settings.broker_producer,
            expected_value=[URL('foo://a'), URL('foo://b'), URL('foo://c')],
        ),
        EnvCase(
            env={'BROKER_CONSUMER_URL': 'foo://a;foo://b;foo://c'},
            setting=Settings.broker_consumer,
            expected_value=[URL('foo://a'), URL('foo://b'), URL('foo://c')],
        ),
        EnvCase(
            env={'BROKER_PRODUCER_URL': 'foo://a;foo://b;foo://c'},
            setting=Settings.broker_producer,
            expected_value=[URL('foo://a'), URL('foo://b'), URL('foo://c')],
        ),
        EnvCase(
            env={'BROKER_API_VERSION': '1.12'},
            setting=Settings.broker_api_version,
            expected_value='1.12',
        ),
        EnvCase(
            env={'BROKER_API_VERSION': '1.12'},
            setting=Settings.consumer_api_version,
            expected_value='1.12',
        ),
        EnvCase(
            env={'CONSUMER_API_VERSION': '1.13'},
            setting=Settings.consumer_api_version,
            expected_value='1.13',
        ),
        EnvCase(
            env={'BROKER_API_VERSION': '1.12'},
            setting=Settings.producer_api_version,
            expected_value='1.12',
        ),
        EnvCase(
            env={'PRODUCER_API_VERSION': '1.14'},
            setting=Settings.producer_api_version,
            expected_value='1.14',
        ),
        EnvCase(
            env={'BROKER_CHECK_CRCS': 'no'},
            setting=Settings.broker_check_crcs,
            expected_value=False,
        ),
        EnvCase(
            env={'BROKER_CLIENT_ID': 'x-y-z'},
            setting=Settings.broker_client_id,
            expected_value='x-y-z',
        ),
        EnvCase(
            env={'BROKER_COMMIT_EVERY': '10'},
            setting=Settings.broker_commit_every,
            expected_value=10,
        ),
        EnvCase(
            env={'BROKER_COMMIT_INTERVAL': '10'},
            setting=Settings.broker_commit_interval,
            expected_value=10.0,
        ),
        EnvCase(
            env={'BROKER_COMMIT_INTERVAL': '10.1234'},
            setting=Settings.broker_commit_interval,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'BROKER_COMMIT_LIVELOCK_SOFT_TIMEOUT': '10.1234'},
            setting=Settings.broker_commit_livelock_soft_timeout,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'BROKER_HEARTBEAT_INTERVAL': '10.1234'},
            setting=Settings.broker_heartbeat_interval,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'BROKER_MAX_POLL_INTERVAL': '10.1234'},
            setting=Settings.broker_max_poll_interval,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'BROKER_MAX_POLL_RECORDS': '30'},
            setting=Settings.broker_max_poll_records,
            expected_value=30,
        ),
        EnvCase(
            env={'BROKER_REBALANCE_TIMEOUT': '10.1234'},
            setting=Settings.broker_rebalance_timeout,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'BROKER_REQUEST_TIMEOUT': '10.1234'},
            setting=Settings.broker_request_timeout,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'BROKER_SESSION_TIMEOUT': '10.1234'},
            setting=Settings.broker_session_timeout,
            expected_value=10.1234,
        ),
        EnvCase(
            env={'CONSUMER_MAX_FETCH_SIZE': '123942012'},
            setting=Settings.consumer_max_fetch_size,
            expected_value=123942012,
        ),
        EnvCase(
            env={'CONSUMER_AUTO_OFFSET_RESET': 'latest'},
            setting=Settings.consumer_auto_offset_reset,
            expected_value='latest',
        ),
        EnvCase(
            env={'APP_KEY_SERIALIZER': 'yaml'},
            setting=Settings.key_serializer,
            expected_value='yaml',
        ),
        EnvCase(
            env={'APP_VALUE_SERIALIZER': 'yaml'},
            setting=Settings.value_serializer,
            expected_value='yaml',
        ),
        EnvCase(
            env={'PRODUCER_ACKS': '0'},
            setting=Settings.producer_acks,
            expected_value=0,
        ),
        EnvCase(
            env={'PRODUCER_ACKS': '1'},
            setting=Settings.producer_acks,
            expected_value=1,
        ),
        EnvCase(
            env={'PRODUCER_ACKS': '-1'},
            setting=Settings.producer_acks,
            expected_value=-1,
        ),
        EnvCase(
            env={'PRODUCER_COMPRESSION_TYPE': 'snappy'},
            setting=Settings.producer_compression_type,
            expected_value='snappy',
        ),
        EnvCase(
            env={'PRODUCER_LINGER_MS': '120392'},
            setting=Settings.producer_linger,
            expected_value=120.392,
        ),
        EnvCase(
            env={'PRODUCER_LINGER': '12.345'},
            setting=Settings.producer_linger,
            expected_value=12.345,
        ),
        EnvCase(
            env={'PRODUCER_MAX_BATCH_SIZE': '120392'},
            setting=Settings.producer_max_batch_size,
            expected_value=120392,
        ),
        EnvCase(
            env={'PRODUCER_MAX_REQUEST_SIZE': '120392'},
            setting=Settings.producer_max_request_size,
            expected_value=120392,
        ),
        EnvCase(
            env={'PRODUCER_REQUEST_TIMEOUT': '120.392'},
            setting=Settings.producer_request_timeout,
            expected_value=120.392,
        ),
        EnvCase(
            env={'APP_REPLY_CREATE_TOPIC': '1'},
            setting=Settings.reply_create_topic,
            expected_value=True,
        ),
        EnvCase(
            env={'APP_REPLY_EXPIRES': '13.321'},
            setting=Settings.reply_expires,
            expected_value=13.321,
        ),
        EnvCase(
            env={'APP_REPLY_TO_PREFIX': 'foo-bar-baz'},
            setting=Settings.reply_to_prefix,
            expected_value='foo-bar-baz',
        ),
        EnvCase(
            env={'PROCESSING_GUARANTEE': 'exactly_once'},
            setting=Settings.processing_guarantee,
            expected_value=ProcessingGuarantee.EXACTLY_ONCE,
        ),
        EnvCase(
            env={'PROCESSING_GUARANTEE': 'at_least_once'},
            setting=Settings.processing_guarantee,
            expected_value=ProcessingGuarantee.AT_LEAST_ONCE,
        ),
        EnvCase(
            env={'STREAM_BUFFER_MAXSIZE': '16384'},
            setting=Settings.stream_buffer_maxsize,
            expected_value=16384,
        ),
        EnvCase(
            env={'STREAM_PROCESSING_TIMEOUT': '12.312'},
            setting=Settings.stream_processing_timeout,
            expected_value=12.312,
        ),
        EnvCase(
            env={'STREAM_RECOVERY_DELAY': '12.312'},
            setting=Settings.stream_recovery_delay,
            expected_value=12.312,
        ),
        EnvCase(
            env={'STREAM_WAIT_EMPTY': 'no'},
            setting=Settings.stream_wait_empty,
            expected_value=False,
        ),
        EnvCase(
            env={'STREAM_WAIT_EMPTY': 'yes'},
            setting=Settings.stream_wait_empty,
            expected_value=True,
        ),
        EnvCase(
            env={'APP_STORE': 'rocksdb://'},
            setting=Settings.store,
            expected_value=URL('rocksdb://'),
        ),
        EnvCase(
            env={'TABLE_CLEANUP_INTERVAL': '60.03'},
            setting=Settings.table_cleanup_interval,
            expected_value=60.03,
        ),
        EnvCase(
            env={'TABLE_KEY_INDEX_SIZE': '32030'},
            setting=Settings.table_key_index_size,
            expected_value=32030,
        ),
        EnvCase(
            env={'TABLE_STANDBY_REPLICAS': '10'},
            setting=Settings.table_standby_replicas,
            expected_value=10,
        ),
        EnvCase(
            env={'TOPIC_ALLOW_DECLARE': '0'},
            setting=Settings.topic_allow_declare,
            expected_value=False,
        ),
        EnvCase(
            env={'TOPIC_ALLOW_DECLARE': '1'},
            setting=Settings.topic_allow_declare,
            expected_value=True,
        ),
        EnvCase(
            env={'TOPIC_DISABLE_LEADER': '0'},
            setting=Settings.topic_disable_leader,
            expected_value=False,
        ),
        EnvCase(
            env={'TOPIC_DISABLE_LEADER': '1'},
            setting=Settings.topic_disable_leader,
            expected_value=True,
        ),
        EnvCase(
            env={'TOPIC_PARTITIONS': '100'},
            setting=Settings.topic_partitions,
            expected_value=100,
        ),
        EnvCase(
            env={'TOPIC_REPLICATION_FACTOR': '100'},
            setting=Settings.topic_replication_factor,
            expected_value=100,
        ),
        EnvCase(
            env={'CACHE_URL': 'redis://'},
            setting=Settings.cache,
            expected_value=URL('redis://'),
        ),
        EnvCase(
            env={'WEB_BIND': '0.0.0.0'},
            setting=Settings.web_bind,
            expected_value='0.0.0.0',
        ),
        EnvCase(
            env={'APP_WEB_ENABLED': 'no'},
            setting=Settings.web_enabled,
            expected_value=False,
        ),
        EnvCase(
            env={'WEB_HOST': 'foo.bar.com'},
            setting=Settings.web_host,
            expected_value='foo.bar.com',
        ),
        EnvCase(
            env={'WEB_HOST': 'foo.bar.com'},
            setting=Settings.canonical_url,
            expected_value=URL('http://foo.bar.com:6066'),
        ),
        EnvCase(
            env={'WEB_PORT': '303'},
            setting=Settings.web_port,
            expected_value=303,
        ),
        EnvCase(
            env={'WEB_PORT': '303', 'WEB_HOST': 'foo.bar.com'},
            setting=Settings.canonical_url,
            expected_value=URL('http://foo.bar.com:303'),
        ),
        EnvCase(
            env={'WORKER_REDIRECT_STDOUTS': 'no'},
            setting=Settings.worker_redirect_stdouts,
            expected_value=False,
        ),
        EnvCase(
            env={'WORKER_REDIRECT_STDOUTS_LEVEL': 'error'},
            setting=Settings.worker_redirect_stdouts_level,
            expected_value='error',
        ),
    ])
    def test_env(self, env: Mapping[str, str], setting: Param[Any], expected_value: Any) -> None:
        app = self.App(env=env)
        self.assert_expected(setting.__get__(app.conf), expected_value)

        # env prefix passed as argument
        prefixed_env = {'FOO_' + k: v for k, v in env.items()}
        app2 = self.App(env=prefixed_env, env_prefix='FOO_')
        self.assert_expected(setting.__get__(app2.conf), expected_value)

        # env prefix set in ENV
        prefixed_env2 = {'BAR_' + k: v for k, v in env.items()}
        prefixed_env2['APP_ENV_PREFIX'] = 'BAR_'
        app3 = self.App(env=prefixed_env2)
        assert app3.conf.env_prefix == 'BAR_'
        self.assert_expected(setting.__get__(app3.conf), expected_value)

    def assert_expected(self, value: Any, expected_value: Any) -> None:
        if expected_value is None:
            assert value is None
        elif expected_value is True:
            assert value is True
        elif expected_value is False:
            assert value is False
        else:
            assert value == expected_value

    def test_defaults(self) -> None:
        app = self.App()
        conf = app.conf
        assert not conf.debug
        assert conf.broker == [URL(Settings.DEFAULT_BROKER_URL)]
        assert conf.broker_consumer == [URL(Settings.DEFAULT_BROKER_URL)]
        assert conf.broker_producer == [URL(Settings.DEFAULT_BROKER_URL)]
        assert conf.store == URL(Settings.store.default)
        assert conf.cache == URL(Settings.cache.default)
        assert conf.web == URL(Settings.web.default)
        assert conf.web_enabled
        assert not conf.web_in_thread
        assert conf.datadir ==