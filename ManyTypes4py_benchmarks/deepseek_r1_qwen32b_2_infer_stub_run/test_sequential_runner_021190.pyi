from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from kedro.io import DataCatalog, DatasetError, MemoryDataset
from kedro.pipeline import modular_pipeline
from pytest import LogCaptureFixture
import pandas as pd

class TestValidSequentialRunner:
    def test_run_with_plugin_manager(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...
    
    def test_run_without_plugin_manager(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        ...
    
    def test_log_not_using_async(self, fan_out_fan_in: Any, catalog: DataCatalog, caplog: LogCaptureFixture) -> None:
        ...
    
    def test_run_twice_giving_same_result(self, fan_out_fan_in: Any, catalog: DataCatalog, hook_manager: Any) -> None:
        ...

class TestSequentialRunnerBranchlessPipeline:
    def test_no_input_seq(self, is_async: bool, branchless_no_input_pipeline: Any, catalog: DataCatalog) -> None:
        ...
    
    def test_no_datasets(self, is_async: bool, branchless_pipeline: Any) -> None:
        ...
    
    def test_no_feed(self, is_async: bool, memory_catalog: DataCatalog, branchless_pipeline: Any) -> None:
        ...
    
    def test_node_returning_none(self, is_async: bool, saving_none_pipeline: Any, catalog: DataCatalog) -> None:
        ...
    
    def test_result_saved_not_returned(self, is_async: bool, saving_result_pipeline: Any) -> None:
        ...

class TestSequentialRunnerBranchedPipeline:
    def test_input_seq(self, is_async: bool, memory_catalog: DataCatalog, unfinished_outputs_pipeline: Any, pandas_df_feed_dict: Dict[str, Any]) -> None:
        ...
    
    def test_conflict_feed_catalog(self, is_async: bool, memory_catalog: DataCatalog, unfinished_outputs_pipeline: Any, conflicting_feed_dict: Dict[str, Any]) -> None:
        ...
    
    def test_unsatisfied_inputs(self, is_async: bool, unfinished_outputs_pipeline: Any, catalog: DataCatalog) -> None:
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
    
    def test_confirms(self, mocker: Any, test_pipeline: Any, is_async: bool) -> None:
        ...

class TestSuggestResumeScenario:
    def test_suggest_resume_scenario(self, caplog: LogCaptureFixture, two_branches_crossed_pipeline: Any, persistent_dataset_catalog: DataCatalog, failing_node_names: List[str], expected_pattern: str) -> None:
        ...
    
    def test_stricter_suggest_resume_scenario(self, caplog: LogCaptureFixture, two_branches_crossed_pipeline_variable_inputs: Any, persistent_dataset_catalog: DataCatalog, failing_node_names: List[str], expected_pattern: str) -> None:
        ...

class TestMemoryDatasetBehaviour:
    def test_run_includes_memory_datasets(self, pipeline_with_memory_datasets: Any) -> None:
        ...