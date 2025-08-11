import datetime
from unittest import mock
import pytest
from dbt.task.freshness import FreshnessResponse, FreshnessTask

class TestFreshnessTaskMetadataCache:

    @pytest.fixture(scope='class')
    def args(self):
        mock_args = mock.Mock()
        mock_args.state = None
        mock_args.defer_state = None
        mock_args.write_json = None
        return mock_args

    @pytest.fixture(scope='class')
    def config(self) -> None:
        mock_config = mock.Mock()
        mock_config.threads = 1
        mock_config.target_name = 'mock_config_target_name'

    @pytest.fixture(scope='class')
    def manifest(self):
        return mock.Mock()

    @pytest.fixture(scope='class')
    def source_with_loaded_at_field(self):
        mock_source = mock.Mock()
        mock_source.unique_id = 'source_with_loaded_at_field'
        mock_source.loaded_at_field = 'loaded_at_field'
        return mock_source

    @pytest.fixture(scope='class')
    def source_no_loaded_at_field(self):
        mock_source = mock.Mock()
        mock_source.unique_id = 'source_no_loaded_at_field'
        return mock_source

    @pytest.fixture(scope='class')
    def source_no_loaded_at_field2(self):
        mock_source = mock.Mock()
        mock_source.unique_id = 'source_no_loaded_at_field2'
        return mock_source

    @pytest.fixture(scope='class')
    def adapter(self):
        return mock.Mock()

    @pytest.fixture(scope='class')
    def freshness_response(self) -> FreshnessResponse:
        return FreshnessResponse(max_loaded_at=datetime.datetime(2020, 5, 2), snapshotted_at=datetime.datetime(2020, 5, 4), age=2)

    def test_populate_metadata_freshness_cache(self, args: Any, config: Union[dict, str], manifest: Union[dict, str], adapter: Any, source_no_loaded_at_field: Union[str, dict], freshness_response: Any) -> None:
        manifest.sources = {source_no_loaded_at_field.unique_id: source_no_loaded_at_field}
        adapter.Relation.create_from.return_value = 'source_relation'
        adapter.calculate_freshness_from_metadata_batch.return_value = ([], {'source_relation': freshness_response})
        task = FreshnessTask(args=args, config=config, manifest=manifest)
        task.populate_metadata_freshness_cache(adapter, {source_no_loaded_at_field.unique_id})
        assert task.get_freshness_metadata_cache() == {'source_relation': freshness_response}

    def test_populate_metadata_freshness_cache_multiple_sources(self, args: Any, config: Union[str, bool, dict], manifest: Union[str, None, dict[str, typing.Any], bool], adapter: Union[bool, None], source_no_loaded_at_field: Union[list[str], bool, str], source_no_loaded_at_field2: Union[str, bool], freshness_response: Union[typing.Mapping, bool]) -> None:
        manifest.sources = {source_no_loaded_at_field.unique_id: source_no_loaded_at_field, source_no_loaded_at_field2.unique_id: source_no_loaded_at_field2}
        adapter.Relation.create_from.side_effect = ['source_relation1', 'source_relation2']
        adapter.calculate_freshness_from_metadata_batch.return_value = ([], {'source_relation1': freshness_response, 'source_relation2': freshness_response})
        task = FreshnessTask(args=args, config=config, manifest=manifest)
        task.populate_metadata_freshness_cache(adapter, {source_no_loaded_at_field.unique_id})
        assert task.get_freshness_metadata_cache() == {'source_relation1': freshness_response, 'source_relation2': freshness_response}

    def test_populate_metadata_freshness_cache_with_loaded_at_field(self, args: Any, config: Union[str, dict, typing.Callable], manifest: Union[dict, str], adapter: Union[bool, None], source_with_loaded_at_field: Union[str, dict], freshness_response: Any) -> None:
        manifest.sources = {source_with_loaded_at_field.unique_id: source_with_loaded_at_field}
        adapter.Relation.create_from.return_value = 'source_relation'
        adapter.calculate_freshness_from_metadata_batch.return_value = ([], {'source_relation': freshness_response})
        task = FreshnessTask(args=args, config=config, manifest=manifest)
        task.populate_metadata_freshness_cache(adapter, {source_with_loaded_at_field.unique_id})
        assert task.get_freshness_metadata_cache() == {'source_relation': freshness_response}

    def test_populate_metadata_freshness_cache_multiple_sources_mixed(self, args: Any, config: Union[bool, dict, dict[str, typing.Any]], manifest: Union[str, None, dict[str, typing.Any], bool], adapter: Union[bool, str, dict], source_no_loaded_at_field: Union[dict, list[str], bool], source_with_loaded_at_field: Union[dict, bool, list[str]], freshness_response: Union[typing.Mapping, bool]) -> None:
        manifest.sources = {source_no_loaded_at_field.unique_id: source_no_loaded_at_field, source_with_loaded_at_field.unique_id: source_with_loaded_at_field}
        adapter.Relation.create_from.return_value = 'source_relation'
        adapter.calculate_freshness_from_metadata_batch.return_value = ([], {'source_relation': freshness_response})
        task = FreshnessTask(args=args, config=config, manifest=manifest)
        task.populate_metadata_freshness_cache(adapter, {source_no_loaded_at_field.unique_id})
        assert task.get_freshness_metadata_cache() == {'source_relation': freshness_response}

    def test_populate_metadata_freshness_cache_adapter_exception(self, args: Any, config: Union[str, typing.Callable, dict], manifest: Union[dict, str, tuple], adapter: Union[bool, None], source_no_loaded_at_field: Union[str, dict], freshness_response: Union[typing.Sequence[str], str, dict[str, typing.Any]]) -> None:
        manifest.sources = {source_no_loaded_at_field.unique_id: source_no_loaded_at_field}
        adapter.Relation.create_from.return_value = 'source_relation'
        adapter.calculate_freshness_from_metadata_batch.side_effect = Exception()
        task = FreshnessTask(args=args, config=config, manifest=manifest)
        task.populate_metadata_freshness_cache(adapter, {source_no_loaded_at_field.unique_id})
        assert task.get_freshness_metadata_cache() == {}