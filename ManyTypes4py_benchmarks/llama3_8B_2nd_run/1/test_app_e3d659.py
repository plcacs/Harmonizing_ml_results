from typing import Any, Mapping, NamedTuple
import faust
import mode
import pytest
from pathlib import Path
from yarl import URL
from faust import App, BootStrategy, LeaderAssignor, PartitionAssignor
from faust.app import Router
from faust.sensors import Monitor
from faust.serializers import Registry
from faust.tables import TableManager
from faust.transport.utils import DefaultSchedulingStrategy
from faust.types.enums import ProcessingGuarantee
from faust.types.settings import Settings, Param
from faust.types.web import ResourceOptions

class EnvCase(NamedTuple):
    pass

class test_settings:
    def App(self, id='myid', **kwargs):
        app = App(id, **kwargs)
        app.finalize()
        return app

    def test_env_with_prefix(self, env: Mapping[str, str], setting: Any, expected_value: Any):
        app = self.App(env=env)
        self.assert_expected(setting.__get__(app.conf), expected_value)
        prefixed_env = {f'FOO_{k}': v for k, v in env.items()}
        app2 = self.App(env=prefixed_env, env_prefix='FOO_')
        self.assert_expected(setting.__get__(app2.conf), expected_value)
        prefixed_env2 = {f'BAR_{k}': v for k, v in env.items()}
        prefixed_env2['APP_ENV_PREFIX'] = 'BAR_'
        app3 = self.App(env=prefixed_env2)
        assert app3.conf.env_prefix == 'BAR_'
        self.assert_expected(setting.__get__(app3.conf), expected_value)

    def assert_expected(self, value: Any, expected_value: Any):
        if expected_value is None:
            assert value is None
        elif expected_value is True:
            assert value is True
        elif expected_value is False:
            assert value is False
        else:
            assert value == expected_value

    def test_defaults(self):
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
        assert conf.datadir == Path('myid-data')
        assert conf.tabledir == Path('myid-data/v1/tables')
        assert conf.blocking_timeout is None
        assert conf.processing_guarantee == ProcessingGuarantee.AT_LEAST_ONCE
        assert conf.broker_api_version == Settings.broker_api_version.default
        assert conf.broker_client_id == Settings.broker_client_id.default
        assert conf.broker_request_timeout == Settings.broker_request_timeout.default
        assert conf.broker_session_timeout == Settings.broker_session_timeout.default
        assert conf.broker_rebalance_timeout == Settings.broker_rebalance_timeout.default
        assert conf.broker_heartbeat_interval == Settings.broker_heartbeat_interval.default
        assert conf.broker_commit_interval == Settings.broker_commit_interval.default
        assert conf.broker_commit_every == Settings.broker_commit_every.default
        assert conf.broker_commit_livelock_soft_timeout == Settings.broker_commit_livelock_soft_timeout.default
        assert conf.broker_check_crcs
        assert conf.consumer_api_version == Settings.broker_api_version.default
        assert conf.timezone is Settings.timezone.default
        assert conf.table_cleanup_interval == Settings.table_cleanup_interval.default
        assert conf.table_key_index_size == Settings.table_key_index_size.default
        assert conf.reply_to_prefix == Settings.reply_to_prefix.default
        assert conf.reply_expires == Settings.reply_expires.default
        assert conf.stream_buffer_maxsize == Settings.stream_buffer_maxsize.default
        assert conf.stream_wait_empty
        assert conf.stream_publish_on_commit
        assert conf.stream_recovery_delay == Settings.stream_recovery_delay.default
        assert