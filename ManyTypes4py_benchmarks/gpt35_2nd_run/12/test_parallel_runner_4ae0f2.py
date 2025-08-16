from __future__ import annotations
from typing import Any
import pytest
from kedro.framework.hooks import _create_hook_manager
from kedro.io import AbstractDataset, DataCatalog, DatasetError, LambdaDataset, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import ParallelRunner
from kedro.runner.parallel_runner import ParallelRunnerManager
from kedro.runner.runner import _MAX_WINDOWS_WORKERS
from tests.runner.conftest import exception_fn, identity, return_none, return_not_serialisable, sink, source

class SingleProcessDataset(AbstractDataset):

    def __init__(self) -> None:
        self._SINGLE_PROCESS: bool = True

    def _load(self) -> None:
        pass

    def _save(self) -> None:
        pass

    def _describe(self) -> None:
        pass

class TestValidParallelRunner:

    @pytest.mark.parametrize('is_async', [False, True])
    def test_parallel_run(self, is_async: bool, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    @pytest.mark.parametrize('is_async', [False, True])
    def test_parallel_run_with_plugin_manager(self, is_async: bool, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    @pytest.mark.parametrize('is_async', [False, True])
    def test_memory_dataset_input(self, is_async: bool, fan_out_fan_in: Any) -> None:
        ...

    def test_log_not_using_async(self, fan_out_fan_in: Any, catalog: DataCatalog, caplog: Any) -> None:
        ...

class TestMaxWorkers:

    @pytest.mark.parametrize('is_async', [False, True])
    @pytest.mark.parametrize('cpu_cores, user_specified_number, expected_number', [(4, 6, 3), (4, None, 3), (2, None, 2), (1, 2, 2)])
    def test_specified_max_workers_bellow_cpu_cores_count(self, is_async: bool, mocker: Any, fan_out_fan_in: Any, catalog: DataCatalog, cpu_cores: int, user_specified_number: int, expected_number: int) -> None:
        ...

    def test_max_worker_windows(self, mocker: Any) -> None:
        ...

@pytest.mark.parametrize('is_async', [False, True])
class TestInvalidParallelRunner:

    def test_task_node_validation(self, is_async: bool, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    def test_task_dataset_validation(self, is_async: bool, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    def test_task_exception(self, is_async: bool, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    def test_memory_dataset_output(self, is_async: bool, fan_out_fan_in: Any) -> None:
        ...

    def test_node_returning_none(self, is_async: bool) -> None:
        ...

    def test_dataset_not_serialisable(self, is_async: bool, fan_out_fan_in: Any) -> None:
        ...

    def test_memory_dataset_not_serialisable(self, is_async: bool, catalog: DataCatalog) -> None:
        ...

    def test_unable_to_schedule_all_nodes(self, mocker: Any, is_async: bool, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

class LoggingDataset(AbstractDataset):

    def __init__(self, log: Any, name: str, value: Any = None) -> None:
        ...

    def _load(self) -> Any:
        ...

    def _save(self, data: Any) -> None:
        ...

    def _release(self) -> None:
        ...

    def _describe(self) -> dict:
        ...

ParallelRunnerManager.register('LoggingDataset', LoggingDataset)

@pytest.fixture
def logging_dataset_catalog() -> DataCatalog:
    ...

@pytest.mark.parametrize('is_async', [False, True])
class TestParallelRunnerRelease:

    def test_dont_release_inputs_and_outputs(self, is_async: bool) -> None:
        ...

    def test_release_at_earliest_opportunity(self, is_async: bool) -> None:
        ...

    def test_count_multiple_loads(self, is_async: bool) -> None:
        ...

    def test_release_transcoded(self, is_async: bool) -> None:
        ...

class TestSuggestResumeScenario:

    @pytest.mark.parametrize('failing_node_names,expected_pattern', [(['node1_A', 'node1_B'], 'No nodes ran.'), (['node2'], '(node1_A,node1_B|node1_B,node1_A)'), (['node3_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'), (['node4_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'), (['node3_A', 'node4_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'), (['node2', 'node4_A'], '(node1_A,node1_B)')])
    def test_suggest_resume_scenario(self, caplog: Any, two_branches_crossed_pipeline: Any, logging_dataset_catalog: DataCatalog, failing_node_names: list, expected_pattern: str) -> None:
        ...

    @pytest.mark.parametrize('failing_node_names,expected_pattern', [(['node1_A', 'node1_B'], 'No nodes ran.'), (['node2'], '"node1_A,node1_B"'), (['node3_A'], '(node3_A,node3_B|node3_A)'), (['node4_A'], '(node3_A,node3_B|node3_A)'), (['node3_A', 'node4_A'], '(node3_A,node3_B|node3_A)'), (['node2', 'node4_A'], '"node1_A,node1_B"')])
    def test_stricter_suggest_resume_scenario(self, caplog: Any, two_branches_crossed_pipeline_variable_inputs: Any, logging_dataset_catalog: DataCatalog, failing_node_names: list, expected_pattern: str) -> None:
        ...
