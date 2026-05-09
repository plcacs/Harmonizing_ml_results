import os
from argparse import Namespace
from datetime import datetime
from typing import Any, Optional, List, Tuple
from unittest import mock
import pytest
import pytz
from pytest_mock import MockerFixture
from dbt.adapters.base import BaseRelation
from dbt.artifacts.resources import NodeConfig, Quoting, SeedConfig
from dbt.artifacts.resources.types import BatchSize
from dbt.context.providers import BaseResolver, EventTimeFilter, RuntimeRefResolver, RuntimeSourceResolver
from dbt.contracts.graph.nodes import BatchContext, ModelNode
from dbt.event_time.sample_window import SampleWindow

class TestBaseResolver:
    class ResolverSubclass(BaseResolver):
        def __call__(self, *args: Any) -> None:
            pass

    @pytest.fixture
    def resolver(self) -> BaseResolver:
        return self.ResolverSubclass(db_wrapper=mock.Mock(), model=mock.Mock(), config=mock.Mock(), manifest=mock.Mock())

    @pytest.mark.parametrize(
        'empty,expected_resolve_limit',
        [
            (False, None),
            (True, 0)
        ]
    )
    def test_resolve_limit(self, resolver: BaseResolver, empty: bool, expected_resolve_limit: Optional[int]) -> None:
        resolver.config.args.EMPTY = empty
        assert resolver.resolve_limit == expected_resolve_limit

    @pytest.mark.parametrize(
        'use_microbatch_batches,materialized,incremental_strategy,sample_mode_available,sample,resolver_model_node,target_type,expect_filter',
        [
            (True, 'incremental', 'microbatch', True, None, True, NodeConfig, True),
            (True, 'incremental', 'microbatch', True, SampleWindow(start=datetime(2024, 1, 1, tzinfo=pytz.UTC), end=datetime(2025, 1, 1, tzinfo=pytz.UTC)), True, NodeConfig, True),
            (False, 'table', None, True, SampleWindow(start=datetime(2024, 1, 1, tzinfo=pytz.UTC), end=datetime(2025, 1, 1, tzinfo=pytz.UTC)), True, NodeConfig, True),
            (True, 'incremental', 'merge', True, SampleWindow(start=datetime(2024, 1, 1, tzinfo=pytz.UTC), end=datetime(2025, 1, 1, tzinfo=pytz.UTC)), True, NodeConfig, True),
            (False, 'table', None, False, SampleWindow(start=datetime(2024, 1, 1, tzinfo=pytz.UTC), end=datetime(2025, 1, 1, tzinfo=pytz.UTC)), True, NodeConfig, False),
            (False, 'table', None, True, SampleWindow(start=datetime(2024, 1, 1, tzinfo=pytz.UTC), end=datetime(2025, 1, 1, tzinfo=pytz.UTC)), False, NodeConfig, False),
            (True, 'incremental', 'microbatch', False, None, False, NodeConfig, False),
            (False, 'incremental', 'microbatch', False, None, True, NodeConfig, False),
            (True, 'table', 'microbatch', False, None, True, NodeConfig, False),
            (True, 'incremental', 'merge', False, None, True, NodeConfig, False),
            (False, 'table', None, True, SampleWindow.from_relative_string('2 days'), True, SeedConfig, True),
            (False, 'table', None, False, SampleWindow.from_relative_string('2 days'), True, SeedConfig, False),
            (False, 'table', None, True, None, True, SeedConfig, False)
        ]
    )
    def test_resolve_event_time_filter(
        self,
        mocker: MockerFixture,
        resolver: BaseResolver,
        use_microbatch_batches: bool,
        materialized: str,
        incremental_strategy: str,
        sample_mode_available: bool,
        sample: Any,
        resolver_model_node: bool,
        target_type: Any,
        expect_filter: bool
    ) -> None:
        target = mock.Mock()
        target.config = mock.MagicMock(target_type)
        target.config.event_time = 'created_at'
        if sample_mode_available:
            mocker.patch.dict(os.environ, {'DBT_EXPERIMENTAL_SAMPLE_MODE': '1'})
        resolver.config.args.EVENT_TIME_END = None
        resolver.config.args.EVENT_TIME_START = None
        resolver.config.args.sample = sample
        if resolver_model_node:
            resolver.model = mock.MagicMock(spec=ModelNode)
        resolver.model.batch = BatchContext(id='1', event_time_start=datetime(2024, 1, 1, tzinfo=pytz.UTC), event_time_end=datetime(2025, 1, 1, tzinfo=pytz.UTC))
        resolver.model.config = mock.MagicMock(NodeConfig)
        resolver.model.config.materialized = materialized
        resolver.model.config.incremental_strategy = incremental_strategy
        resolver.model.config.batch_size = BatchSize.day
        resolver.model.config.lookback = 1
        resolver.manifest.use_microbatch_batches = mock.Mock()
        resolver.manifest.use_microbatch_batches.return_value = use_microbatch_batches
        event_time_filter = resolver.resolve_event_time_filter(target=target)
        if expect_filter:
            assert isinstance(event_time_filter, EventTimeFilter)
        else:
            assert event_time_filter is None

class TestRuntimeRefResolver:

    @pytest.fixture
    def resolver(self) -> RuntimeRefResolver:
        mock_db_wrapper = mock.Mock()
        mock_db_wrapper.Relation = BaseRelation
        return RuntimeRefResolver(db_wrapper=mock_db_wrapper, model=mock.Mock(), config=mock.Mock(), manifest=mock.Mock())

    @pytest.mark.parametrize(
        'empty,is_ephemeral_model,expected_limit',
        [
            (False, False, None),
            (True, False, 0),
            (False, True, None),
            (True, True, 0)
        ]
    )
    def test_create_relation_with_empty(
        self,
        resolver: RuntimeRefResolver,
        empty: bool,
        is_ephemeral_model: bool,
        expected_limit: Optional[int]
    ) -> None:
        resolver.config.args.EMPTY = empty
        resolver.config.quoting = {}
        mock_node = mock.Mock()
        mock_node.database = 'test'
        mock_node.schema = 'test'
        mock_node.identifier = 'test'
        mock_node.quoting_dict = {}
        mock_node.alias = 'test'
        mock_node.is_ephemeral_model = is_ephemeral_model
        mock_node.defer_relation = None
        set_from_args(Namespace(require_batched_execution_for_custom_microbatch_strategy=False), None)
        with mock.patch('dbt.contracts.graph.nodes.ParsedNode', new=mock.Mock):
            relation = resolver.create_relation(mock_node)
        assert relation.limit == expected_limit

class TestRuntimeSourceResolver:

    @pytest.fixture
    def resolver(self) -> RuntimeSourceResolver:
        mock_db_wrapper = mock.Mock()
        mock_db_wrapper.Relation = BaseRelation
        return RuntimeSourceResolver(db_wrapper=mock_db_wrapper, model=mock.Mock(), config=mock.Mock(), manifest=mock.Mock())

    @pytest.mark.parametrize(
        'empty,expected_limit',
        [
            (False, None),
            (True, 0)
        ]
    )
    def test_create_relation_with_empty(
        self,
        resolver: RuntimeSourceResolver,
        empty: bool,
        expected_limit: Optional[int]
    ) -> None:
        resolver.config.args.EMPTY = empty
        resolver.config.quoting = {}
        mock_source = mock.Mock()
        mock_source.database = 'test'
        mock_source.schema = 'test'
        mock_source.identifier = 'test'
        mock_source.quoting = Quoting()
        mock_source.quoting_dict = {}
        resolver.manifest.resolve_source.return_value = mock_source
        set_from_args(Namespace(require_batched_execution_for_custom_microbatch_strategy=False), None)
        relation = resolver.resolve('test', 'test')
        assert relation.limit == expected_limit
