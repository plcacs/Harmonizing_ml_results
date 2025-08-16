from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
import pytest
from kedro.framework.hooks import _create_hook_manager
from kedro.io import AbstractDataset, DataCatalog, DatasetError, LambdaDataset, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import SequentialRunner

class TestValidSequentialRunner:

    def test_run_with_plugin_manager(self, fan_out_fan_in, catalog: DataCatalog) -> None:
        ...

    def test_run_without_plugin_manager(self, fan_out_fan_in, catalog: DataCatalog) -> None:
        ...

    def test_log_not_using_async(self, fan_out_fan_in, catalog: DataCatalog, caplog) -> None:
        ...

    def test_run_twice_giving_same_result(self, fan_out_fan_in, catalog: DataCatalog) -> None:
        ...

class TestSeqentialRunnerBranchlessPipeline:

    def test_no_input_seq(self, is_async: bool, branchless_no_input_pipeline, catalog: DataCatalog) -> None:
        ...

    def test_no_datasets(self, is_async: bool, branchless_pipeline) -> None:
        ...

    def test_no_feed(self, is_async: bool, memory_catalog: DataCatalog, branchless_pipeline) -> None:
        ...

    def test_node_returning_none(self, is_async: bool, saving_none_pipeline, catalog: DataCatalog) -> None:
        ...

    def test_result_saved_not_returned(self, is_async: bool, saving_result_pipeline) -> None:
        ...

class TestSequentialRunnerBranchedPipeline:

    def test_input_seq(self, is_async: bool, memory_catalog: DataCatalog, unfinished_outputs_pipeline, pandas_df_feed_dict: Dict[str, Any]) -> None:
        ...

    def test_conflict_feed_catalog(self, is_async: bool, memory_catalog: DataCatalog, unfinished_outputs_pipeline, conflicting_feed_dict: Dict[str, Any]) -> None:
        ...

    def test_unsatisfied_inputs(self, is_async: bool, unfinished_outputs_pipeline, catalog: DataCatalog) -> None:
        ...

class LoggingDataset(AbstractDataset):

    def __init__(self, log: List[Tuple[str, str]], name: str, value: Any = None) -> None:
        ...

    def _load(self) -> Any:
        ...

    def _save(self, data: Any) -> None:
        ...

    def _release(self) -> None:
        ...

    def _describe(self) -> Dict[str, Any]:
        ...

class TestSequentialRunnerRelease:

    def test_dont_release_inputs_and_outputs(self, is_async: bool) -> None:
        ...

    def test_release_at_earliest_opportunity(self, is_async: bool) -> None:
        ...

    def test_count_multiple_loads(self, is_async: bool) -> None:
        ...

    def test_release_transcoded(self, is_async: bool) -> None:
        ...

    def test_confirms(self, mocker, test_pipeline: modular_pipeline, is_async: bool) -> None:
        ...

class TestSuggestResumeScenario:

    def test_suggest_resume_scenario(self, caplog, two_branches_crossed_pipeline, persistent_dataset_catalog: DataCatalog, failing_node_names: List[str], expected_pattern: str) -> None:
        ...

    def test_stricter_suggest_resume_scenario(self, caplog, two_branches_crossed_pipeline_variable_inputs, persistent_dataset_catalog: DataCatalog, failing_node_names: List[str], expected_pattern: str) -> None:
        ...

class TestMemoryDatasetBehaviour:

    def test_run_includes_memory_datasets(self, pipeline_with_memory_datasets) -> None:
        ...
