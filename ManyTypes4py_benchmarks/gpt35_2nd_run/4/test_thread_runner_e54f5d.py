from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import pytest
from kedro.framework.hooks import _create_hook_manager
from kedro.io import AbstractDataset, DataCatalog, DatasetError, KedroDataCatalog, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import ThreadRunner
from tests.runner.conftest import exception_fn, identity, return_none, sink, source

class TestValidThreadRunner:

    def test_thread_run(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    def test_thread_run_with_plugin_manager(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    def test_memory_dataset_input(self, fan_out_fan_in: Any) -> None:
        ...

    def test_does_not_log_not_using_async(self, fan_out_fan_in: Any, catalog: DataCatalog, caplog: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_thread_run_with_patterns(self, catalog_type: Any) -> None:
        ...

class TestMaxWorkers:

    @pytest.mark.parametrize('user_specified_number, expected_number', [(6, 3), (None, 3)])
    def test_specified_max_workers(self, mocker: Any, fan_out_fan_in: Any, catalog: DataCatalog, user_specified_number: int, expected_number: int) -> None:
        ...

    def test_init_with_negative_process_count(self) -> None:
        ...

class TestIsAsync:

    def test_thread_run(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

class TestInvalidThreadRunner:

    def test_task_exception(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...

    def test_node_returning_none(self) -> None:
        ...

class LoggingDataset(AbstractDataset):

    def __init__(self, log: list, name: str, value: Any = None) -> None:
        ...

    def _load(self) -> Any:
        ...

    def _save(self, data: Any) -> None:
        ...

    def _release(self) -> None:
        ...

    def _describe(self) -> dict:
        ...

class TestThreadRunnerRelease:

    def test_dont_release_inputs_and_outputs(self) -> None:
        ...

    def test_release_at_earliest_opportunity(self) -> None:
        ...

    def test_count_multiple_loads(self) -> None:
        ...

    def test_release_transcoded(self) -> None:
        ...

class TestSuggestResumeScenario:

    @pytest.mark.parametrize('failing_node_names,expected_pattern', [(['node1_A', 'node1_B'], 'No nodes ran.'), (['node2'], '(node1_A,node1_B|node1_B,node1_A)'), (['node3_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'), (['node4_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'), (['node3_A', 'node4_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'), (['node2', 'node4_A'], '(node1_A,node1_B|node1_B,node1_A)')])
    def test_suggest_resume_scenario(self, caplog: Any, two_branches_crossed_pipeline: Any, persistent_dataset_catalog: DataCatalog, failing_node_names: list, expected_pattern: str) -> None:
        ...

    @pytest.mark.parametrize('failing_node_names,expected_pattern', [(['node1_A', 'node1_B'], 'No nodes ran.'), (['node2'], '"node1_A,node1_B"'), (['node3_A'], '(node3_A,node3_B|node3_A)'), (['node4_A'], '(node3_A,node3_B|node3_A)'), (['node3_A', 'node4_A'], '(node3_A,node3_B|node3_A)'), (['node2', 'node4_A'], '"node1_A,node1_B"')])
    def test_stricter_suggest_resume_scenario(self, caplog: Any, two_branches_crossed_pipeline_variable_inputs: Any, persistent_dataset_catalog: DataCatalog, failing_node_names: list, expected_pattern: str) -> None:
        ...
