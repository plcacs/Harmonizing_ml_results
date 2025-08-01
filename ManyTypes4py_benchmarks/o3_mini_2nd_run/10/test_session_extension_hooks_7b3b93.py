#!/usr/bin/env python
import logging
import multiprocessing
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import pytest
from dynaconf.validator import Validator
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.framework.hooks import _create_hook_manager, hook_impl
from kedro.framework.hooks.manager import _register_hooks, _register_hooks_entry_points
from kedro.framework.project import _ProjectPipelines, _ProjectSettings, pipelines, settings
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline, Pipeline
from kedro.pipeline.node import Node
from kedro.runner import ParallelRunner
from kedro.runner.task import Task
from tests.framework.session.conftest import (
    _assert_hook_call_record_has_expected_parameters,
    _assert_pipeline_equal,
    _mock_imported_settings_paths,
    assert_exceptions_equal,
)

SKIP_ON_WINDOWS_AND_MACOS = pytest.mark.skipif(
    multiprocessing.get_start_method() == "spawn", reason="Due to bug in parallel runner"
)
logger: logging.Logger = logging.getLogger("tests.framework.session.conftest")
logger.setLevel(logging.DEBUG)


def broken_node() -> None:
    raise ValueError("broken")


@pytest.fixture
def broken_pipeline() -> Pipeline:
    return pipeline(
        [node(broken_node, None, "A", name="node1"), node(broken_node, None, "B", name="node2")],
        tags="pipeline",
    )


@pytest.fixture
def mock_broken_pipelines(mocker: Any, broken_pipeline: Pipeline) -> Callable[[], Dict[str, Pipeline]]:
    def mock_get_pipelines_registry_callable() -> Dict[str, Pipeline]:
        return {"__default__": broken_pipeline}

    mocker.patch.object(_ProjectPipelines, "_get_pipelines_registry_callable", return_value=mock_get_pipelines_registry_callable)
    return mock_get_pipelines_registry_callable


class TestCatalogHooks:
    def test_after_catalog_created_hook(self, mock_session: KedroSession, caplog: Any) -> None:
        context = mock_session.load_context()
        project_path: Any = context.project_path
        catalog: DataCatalog = context.catalog
        config_loader: Dict[str, Any] = mock_session._get_config_loader()  # type: ignore
        relevant_records: List[Any] = [r for r in caplog.records if r.getMessage() == "Catalog created"]
        assert len(relevant_records) == 1
        record = relevant_records[0]
        assert record.catalog is catalog
        assert record.conf_creds == config_loader["credentials"]
        assert record.conf_catalog == _convert_paths_to_absolute_posix(
            project_path=project_path, conf_dictionary=config_loader["catalog"]
        )
        assert record.save_version is None
        assert record.load_versions is None

    def test_after_catalog_created_hook_on_session_run(self, mocker: Any, mock_session: KedroSession, dummy_dataframe: pd.DataFrame, caplog: Any) -> None:
        context = mock_session.load_context()
        fake_save_version: Any = mocker.sentinel.fake_save_version
        mocker.patch(
            "kedro.framework.session.KedroSession.store",
            new_callable=mocker.PropertyMock,
            return_value={"session_id": fake_save_version, "save_version": fake_save_version},
        )
        catalog: DataCatalog = context.catalog
        config_loader: Dict[str, Any] = mock_session._get_config_loader()  # type: ignore
        project_path: Any = context.project_path
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        mock_session.run()
        relevant_records: List[Any] = [r for r in caplog.records if r.getMessage() == "Catalog created"]
        assert len(relevant_records) == 2
        record = relevant_records[1]
        assert record.conf_creds == config_loader["credentials"]
        assert record.conf_catalog == _convert_paths_to_absolute_posix(
            project_path=project_path, conf_dictionary=config_loader["catalog"]
        )
        assert record.save_version is fake_save_version
        assert record.load_versions is None


class TestPipelineHooks:
    @pytest.mark.usefixtures("mock_pipelines")
    def test_before_and_after_pipeline_run_hooks(self, caplog: Any, mock_session: KedroSession, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        catalog: DataCatalog = context.catalog
        default_pipeline: Pipeline = pipelines["__default__"]
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        mock_session.run()
        before_pipeline_run_calls: List[Any] = [record for record in caplog.records if record.funcName == "before_pipeline_run"]
        assert len(before_pipeline_run_calls) == 1
        call_record = before_pipeline_run_calls[0]
        _assert_pipeline_equal(call_record.pipeline, default_pipeline)
        _assert_hook_call_record_has_expected_parameters(call_record, ["pipeline", "catalog", "run_params"])
        after_pipeline_run_calls: List[Any] = [record for record in caplog.records if record.funcName == "after_pipeline_run"]
        assert len(after_pipeline_run_calls) == 1
        call_record = after_pipeline_run_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["pipeline", "catalog", "run_params"])
        _assert_pipeline_equal(call_record.pipeline, default_pipeline)

    @pytest.mark.usefixtures("mock_broken_pipelines")
    def test_on_pipeline_error_hook(self, caplog: Any, mock_session: KedroSession) -> None:
        with pytest.raises(ValueError, match="broken"):
            mock_session.run()
        on_pipeline_error_calls: List[Any] = [record for record in caplog.records if record.funcName == "on_pipeline_error"]
        assert len(on_pipeline_error_calls) == 1
        call_record = on_pipeline_error_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["error", "run_params", "pipeline", "catalog"])
        expected_error: ValueError = ValueError("broken")
        assert_exceptions_equal(call_record.error, expected_error)

    @pytest.mark.usefixtures("mock_broken_pipelines")
    def test_on_node_error_hook_sequential_runner(self, caplog: Any, mock_session: KedroSession) -> None:
        with pytest.raises(ValueError, match="broken"):
            mock_session.run(node_names=["node1"])
        on_node_error_calls: List[Any] = [record for record in caplog.records if record.funcName == "on_node_error"]
        assert len(on_node_error_calls) == 1
        call_record = on_node_error_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["error", "node", "catalog", "inputs", "is_async", "session_id"])
        expected_error: ValueError = ValueError("broken")
        assert_exceptions_equal(call_record.error, expected_error)


class TestNodeHooks:
    @pytest.mark.usefixtures("mock_pipelines")
    def test_before_and_after_node_run_hooks_sequential_runner(self, caplog: Any, mock_session: KedroSession, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        mock_session.run(node_names=["node1"])
        before_node_run_calls: List[Any] = [record for record in caplog.records if record.funcName == "before_node_run"]
        assert len(before_node_run_calls) == 1
        call_record = before_node_run_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["node", "catalog", "inputs", "is_async", "session_id"])
        assert call_record.inputs["cars"].to_dict() == dummy_dataframe.to_dict()
        after_node_run_calls: List[Any] = [record for record in caplog.records if record.funcName == "after_node_run"]
        assert len(after_node_run_calls) == 1
        call_record = after_node_run_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["node", "catalog", "inputs", "outputs", "is_async", "session_id"])
        assert call_record.outputs["planes"].to_dict() == dummy_dataframe.to_dict()

    @SKIP_ON_WINDOWS_AND_MACOS
    @pytest.mark.usefixtures("mock_broken_pipelines")
    def test_on_node_error_hook_parallel_runner(self, mock_session: KedroSession, logs_listener: Any) -> None:
        with pytest.raises(ValueError, match="broken"):
            mock_session.run(runner=ParallelRunner(max_workers=1), node_names=["node1", "node2"])
        on_node_error_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "on_node_error"]
        assert len(on_node_error_records) == 2
        for call_record in on_node_error_records:
            _assert_hook_call_record_has_expected_parameters(call_record, ["error", "node", "catalog", "inputs", "is_async", "session_id"])
            expected_error: ValueError = ValueError("broken")
            assert_exceptions_equal(call_record.error, expected_error)

    @SKIP_ON_WINDOWS_AND_MACOS
    @pytest.mark.usefixtures("mock_pipelines")
    def test_before_and_after_node_run_hooks_parallel_runner(self, mock_session: KedroSession, logs_listener: Any, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        mock_session.run(runner=ParallelRunner(), node_names=["node1", "node2"])
        before_node_run_log_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "before_node_run"]
        assert len(before_node_run_log_records) == 2
        for record in before_node_run_log_records:
            assert record.getMessage() == "About to run node"
            assert record.node.name in ["node1", "node2"]
            assert set(record.inputs.keys()) <= {"cars", "boats"}
        after_node_run_log_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "after_node_run"]
        assert len(after_node_run_log_records) == 2
        for record in after_node_run_log_records:
            assert record.getMessage() == "Ran node"
            assert record.node.name in ["node1", "node2"]
            assert set(record.outputs.keys()) <= {"planes", "ships"}


class TestDatasetHooks:
    @pytest.mark.usefixtures("mock_pipelines")
    def test_before_and_after_dataset_loaded_hooks_sequential_runner(self, mock_session: KedroSession, caplog: Any, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        mock_session.run(node_names=["node1"])
        before_dataset_loaded_calls: List[Any] = [record for record in caplog.records if record.funcName == "before_dataset_loaded"]
        assert len(before_dataset_loaded_calls) == 1
        call_record = before_dataset_loaded_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["dataset_name"])
        assert call_record.dataset_name == "cars"
        after_dataset_loaded_calls: List[Any] = [record for record in caplog.records if record.funcName == "after_dataset_loaded"]
        assert len(after_dataset_loaded_calls) == 1
        call_record = after_dataset_loaded_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["dataset_name", "data"])
        assert call_record.dataset_name == "cars"
        pd.testing.assert_frame_equal(call_record.data, dummy_dataframe)

    @SKIP_ON_WINDOWS_AND_MACOS
    @pytest.mark.usefixtures("mock_settings")
    def test_before_and_after_dataset_loaded_hooks_parallel_runner(self, mock_session: KedroSession, logs_listener: Any, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        mock_session.run(runner=ParallelRunner(), node_names=["node1", "node2"])
        before_dataset_loaded_log_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "before_dataset_loaded"]
        assert len(before_dataset_loaded_log_records) == 2
        for record in before_dataset_loaded_log_records:
            assert record.getMessage() == "Before dataset loaded"
            assert record.dataset_name in ["cars", "boats"]
        after_dataset_loaded_log_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "after_dataset_loaded"]
        assert len(after_dataset_loaded_log_records) == 2
        for record in after_dataset_loaded_log_records:
            assert record.getMessage() == "After dataset loaded"
            assert record.dataset_name in ["cars", "boats"]
            pd.testing.assert_frame_equal(record.data, dummy_dataframe)

    def test_before_and_after_dataset_saved_hooks_sequential_runner(self, mock_session: KedroSession, caplog: Any, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        context.catalog.save("cars", dummy_dataframe)
        mock_session.run(node_names=["node1"])
        before_dataset_saved_calls: List[Any] = [record for record in caplog.records if record.funcName == "before_dataset_saved"]
        assert len(before_dataset_saved_calls) == 1
        call_record = before_dataset_saved_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["dataset_name", "data"])
        assert call_record.dataset_name == "planes"
        assert call_record.data.to_dict() == dummy_dataframe.to_dict()
        after_dataset_saved_calls: List[Any] = [record for record in caplog.records if record.funcName == "after_dataset_saved"]
        assert len(after_dataset_saved_calls) == 1
        call_record = after_dataset_saved_calls[0]
        _assert_hook_call_record_has_expected_parameters(call_record, ["dataset_name", "data"])
        assert call_record.dataset_name == "planes"
        assert call_record.data.to_dict() == dummy_dataframe.to_dict()

    @SKIP_ON_WINDOWS_AND_MACOS
    def test_before_and_after_dataset_saved_hooks_parallel_runner(self, mock_session: KedroSession, logs_listener: Any, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        mock_session.run(runner=ParallelRunner(), node_names=["node1", "node2"])
        before_dataset_saved_log_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "before_dataset_saved"]
        assert len(before_dataset_saved_log_records) == 2
        for record in before_dataset_saved_log_records:
            assert record.getMessage() == "Before dataset saved"
            assert record.dataset_name in ["planes", "ships"]
            assert record.data.to_dict() == dummy_dataframe.to_dict()
        after_dataset_saved_log_records: List[Any] = [r for r in logs_listener.logs if r.funcName == "after_dataset_saved"]
        assert len(after_dataset_saved_log_records) == 2
        for record in after_dataset_saved_log_records:
            assert record.getMessage() == "After dataset saved"
            assert record.dataset_name in ["planes", "ships"]
            assert record.data.to_dict() == dummy_dataframe.to_dict()


class MockDatasetReplacement:
    pass


@pytest.fixture
def mock_session_with_before_node_run_hooks(
    mocker: Any, project_hooks: Any, mock_package_name: str, tmp_path: Any
) -> KedroSession:
    class BeforeNodeRunHook:
        """Should overwrite the `cars` dataset"""

        @hook_impl
        def before_node_run(self, node: Node) -> Optional[Dict[str, Any]]:
            return {"cars": MockDatasetReplacement()} if node.name == "node1" else None

    class MockSettings(_ProjectSettings):
        _HOOKS: Validator = Validator("HOOKS", default=(project_hooks, BeforeNodeRunHook()))

    _mock_imported_settings_paths(mocker, MockSettings())
    return KedroSession.create(tmp_path)


@pytest.fixture
def mock_session_with_broken_before_node_run_hooks(
    mocker: Any, project_hooks: Any, mock_package_name: str, tmp_path: Any
) -> KedroSession:
    class BeforeNodeRunHook:
        """Should overwrite the `cars` dataset"""

        @hook_impl
        def before_node_run(self) -> Any:
            return MockDatasetReplacement()

    class MockSettings(_ProjectSettings):
        _HOOKS: Validator = Validator("HOOKS", default=(project_hooks, BeforeNodeRunHook()))

    _mock_imported_settings_paths(mocker, MockSettings())
    return KedroSession.create(tmp_path)


class TestBeforeNodeRunHookWithInputUpdates:
    """Test the behavior of `before_node_run_hook` when updating node inputs."""

    def test_correct_input_update(self, mock_session_with_before_node_run_hooks: KedroSession, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session_with_before_node_run_hooks.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        result: Dict[str, Any] = mock_session_with_before_node_run_hooks.run()  # type: ignore
        assert isinstance(result["planes"], MockDatasetReplacement)
        assert isinstance(result["ships"], pd.DataFrame)

    @SKIP_ON_WINDOWS_AND_MACOS
    def test_correct_input_update_parallel(self, mock_session_with_before_node_run_hooks: KedroSession, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session_with_before_node_run_hooks.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        result: Dict[str, Any] = mock_session_with_before_node_run_hooks.run(runner=ParallelRunner())  # type: ignore
        assert isinstance(result["planes"], MockDatasetReplacement)
        assert isinstance(result["ships"], pd.DataFrame)

    def test_broken_input_update(self, mock_session_with_broken_before_node_run_hooks: KedroSession, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session_with_broken_before_node_run_hooks.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        pattern: str = (
            "'before_node_run' must return either None or a dictionary mapping dataset names to updated values, "
            "got 'MockDatasetReplacement'"
        )
        with pytest.raises(TypeError, match=re.escape(pattern)):
            mock_session_with_broken_before_node_run_hooks.run()  # type: ignore

    @SKIP_ON_WINDOWS_AND_MACOS
    def test_broken_input_update_parallel(self, mock_session_with_broken_before_node_run_hooks: KedroSession, dummy_dataframe: pd.DataFrame) -> None:
        context = mock_session_with_broken_before_node_run_hooks.load_context()
        catalog: DataCatalog = context.catalog
        catalog.save("cars", dummy_dataframe)
        catalog.save("boats", dummy_dataframe)
        pattern: str = (
            "'before_node_run' must return either None or a dictionary mapping dataset names to updated values, "
            "got 'MockDatasetReplacement'"
        )
        with pytest.raises(TypeError, match=re.escape(pattern)):
            mock_session_with_broken_before_node_run_hooks.run(runner=ParallelRunner())  # type: ignore


def wait_and_identity(*args: Any) -> Any:
    time.sleep(0.1)
    if len(args) == 1:
        return args[0]
    return args


@pytest.fixture
def sample_node() -> Node:
    return node(wait_and_identity, inputs="ds1", outputs="ds2", name="test-node")


@pytest.fixture
def sample_node_multiple_outputs() -> Node:
    return node(wait_and_identity, inputs=["ds1", "ds2"], outputs=["ds3", "ds4"], name="test-node")


class LogCatalog(DataCatalog):
    def load(self, name: str, version: Optional[Any] = None) -> Any:
        dataset: Any = super().load(name=name, version=version)
        logger.info("Catalog load")
        return dataset


@pytest.fixture
def memory_catalog() -> LogCatalog:
    ds1: MemoryDataset = MemoryDataset({"data": 42})
    ds2: MemoryDataset = MemoryDataset({"data": 42})
    ds3: MemoryDataset = MemoryDataset({"data": 42})
    ds4: MemoryDataset = MemoryDataset({"data": 42})
    return LogCatalog({"ds1": ds1, "ds2": ds2, "ds3": ds3, "ds4": ds4})


@pytest.fixture
def hook_manager() -> Any:
    hm: Any = _create_hook_manager()
    _register_hooks(hm, settings.HOOKS)
    _register_hooks_entry_points(hm, settings.DISABLE_HOOKS_FOR_PLUGINS)
    return hm


class TestAsyncNodeDatasetHooks:
    @pytest.mark.usefixtures("mock_settings")
    def test_after_dataset_load_hook_async(self, memory_catalog: LogCatalog, mock_session: KedroSession, sample_node: Node, logs_listener: Any) -> None:
        mock_session.load_context()
        task: Task = Task(node=sample_node, catalog=memory_catalog, hook_manager=mock_session._hook_manager, is_async=True)  # type: ignore
        task.execute()
        hooks_log_messages: List[Any] = [r.message for r in logs_listener.logs]
        expected_messages = str(["Before dataset loaded", "Catalog load", "After dataset loaded"]).strip("[]")
        assert expected_messages in str(hooks_log_messages).strip("[]")

    def test_after_dataset_load_hook_async_multiple_outputs(self, mocker: Any, memory_catalog: LogCatalog, hook_manager: Any, sample_node_multiple_outputs: Node) -> None:
        after_dataset_saved_mock = mocker.patch.object(hook_manager.hook, "after_dataset_saved")
        task: Task = Task(node=sample_node_multiple_outputs, catalog=memory_catalog, hook_manager=hook_manager, is_async=True)
        task.execute()
        after_dataset_saved_mock.assert_has_calls(
            [
                mocker.call(dataset_name="ds3", data={"data": 42}, node=sample_node_multiple_outputs),
                mocker.call(dataset_name="ds4", data={"data": 42}, node=sample_node_multiple_outputs),
            ],
            any_order=True,
        )
        assert after_dataset_saved_mock.call_count == 2


class TestKedroContextSpecsHook:
    """Test the behavior of `after_context_created` when updating node inputs."""

    def test_after_context_created_hook(self, mock_session: KedroSession, caplog: Any) -> None:
        context = mock_session.load_context()
        relevant_records: List[Any] = [r for r in caplog.records if r.getMessage() == "After context created"]
        assert len(relevant_records) == 1
        record = relevant_records[0]
        assert record.context is context
