from __future__ import annotations

from typing import Any

import pytest
from kedro.io import AbstractDataset, DataCatalog, KedroDataCatalog
from kedro.pipeline import Pipeline


class TestValidThreadRunner:
    def test_thread_run(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None: ...
    def test_thread_run_with_plugin_manager(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None: ...
    def test_memory_dataset_input(self, fan_out_fan_in: Pipeline) -> None: ...
    def test_does_not_log_not_using_async(self, fan_out_fan_in: Pipeline, catalog: DataCatalog, caplog: pytest.LogCaptureFixture) -> None: ...
    @pytest.mark.parametrize("catalog_type", [DataCatalog, KedroDataCatalog])
    def test_thread_run_with_patterns(self, catalog_type: type[DataCatalog] | type[KedroDataCatalog]) -> None: ...


class TestMaxWorkers:
    @pytest.mark.parametrize("user_specified_number, expected_number", [(6, 3), (None, 3)])
    def test_specified_max_workers(
        self,
        mocker: Any,
        fan_out_fan_in: Pipeline,
        catalog: DataCatalog,
        user_specified_number: int | None,
        expected_number: int,
    ) -> None: ...
    def test_init_with_negative_process_count(self) -> None: ...


class TestIsAsync:
    def test_thread_run(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None: ...


class TestInvalidThreadRunner:
    def test_task_exception(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None: ...
    def test_node_returning_none(self) -> None: ...


class LoggingDataset(AbstractDataset):
    log: list[tuple[str, str]]
    name: str
    value: Any

    def __init__(self, log: list[tuple[str, str]], name: str, value: Any = ...) -> None: ...
    def _load(self) -> Any: ...
    def _save(self, data: Any) -> None: ...
    def _release(self) -> None: ...
    def _describe(self) -> dict[str, Any]: ...


class TestThreadRunnerRelease:
    def test_dont_release_inputs_and_outputs(self) -> None: ...
    def test_release_at_earliest_opportunity(self) -> None: ...
    def test_count_multiple_loads(self) -> None: ...
    def test_release_transcoded(self) -> None: ...


class TestSuggestResumeScenario:
    @pytest.mark.parametrize(
        "failing_node_names,expected_pattern",
        [
            (["node1_A", "node1_B"], "No nodes ran."),
            (["node2"], "(node1_A,node1_B|node1_B,node1_A)"),
            (["node3_A"], "(node3_A,node3_B|node3_B,node3_A|node3_A)"),
            (["node4_A"], "(node3_A,node3_B|node3_B,node3_A|node3_A)"),
            (["node3_A", "node4_A"], "(node3_A,node3_B|node3_B,node3_A|node3_A)"),
            (["node2", "node4_A"], "(node1_A,node1_B|node1_B,node1_A)"),
        ],
    )
    def test_suggest_resume_scenario(
        self,
        caplog: pytest.LogCaptureFixture,
        two_branches_crossed_pipeline: Pipeline,
        persistent_dataset_catalog: DataCatalog,
        failing_node_names: list[str],
        expected_pattern: str,
    ) -> None: ...

    @pytest.mark.parametrize(
        "failing_node_names,expected_pattern",
        [
            (["node1_A", "node1_B"], "No nodes ran."),
            (["node2"], '"node1_A,node1_B"'),
            (["node3_A"], "(node3_A,node3_B|node3_A)"),
            (["node4_A"], "(node3_A,node3_B|node3_A)"),
            (["node3_A", "node4_A"], "(node3_A,node3_B|node3_A)"),
            (["node2", "node4_A"], '"node1_A,node1_B"'),
        ],
    )
    def test_stricter_suggest_resume_scenario(
        self,
        caplog: pytest.LogCaptureFixture,
        two_branches_crossed_pipeline_variable_inputs: Pipeline,
        persistent_dataset_catalog: DataCatalog,
        failing_node_names: list[str],
        expected_pattern: str,
    ) -> None: ...