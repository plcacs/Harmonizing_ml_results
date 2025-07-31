from __future__ import annotations
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import pytest
from kedro.framework.hooks import _create_hook_manager
from kedro.io import AbstractDataset, DataCatalog, DatasetError, LambdaDataset, MemoryDataset
from kedro.pipeline import node, Pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import SequentialRunner
from tests.runner.conftest import exception_fn, identity, sink, source


class TestValidSequentialRunner:
    def test_run_with_plugin_manager(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        catalog.add_feed_dict({"A": 42})
        result: Dict[str, Any] = SequentialRunner().run(fan_out_fan_in, catalog, hook_manager=_create_hook_manager())
        assert "Z" in result
        assert result["Z"] == (42, 42, 42)

    def test_run_without_plugin_manager(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        catalog.add_feed_dict({"A": 42})
        result: Dict[str, Any] = SequentialRunner().run(fan_out_fan_in, catalog)
        assert "Z" in result
        assert result["Z"] == (42, 42, 42)

    def test_log_not_using_async(self, fan_out_fan_in: Any, catalog: DataCatalog, caplog: Any) -> None:
        catalog.add_feed_dict({"A": 42})
        SequentialRunner().run(fan_out_fan_in, catalog)
        assert "Using synchronous mode for loading and saving data." in caplog.text

    def test_run_twice_giving_same_result(self, fan_out_fan_in: Any, catalog: DataCatalog) -> None:
        catalog.add_feed_dict({"A": 42})
        patterns_before_run: List[Any] = catalog.config_resolver.list_patterns()
        result_first_run: Dict[str, Any] = SequentialRunner().run(
            fan_out_fan_in, catalog, hook_manager=_create_hook_manager()
        )
        assert patterns_before_run == catalog.config_resolver.list_patterns()
        result_second_run: Dict[str, Any] = SequentialRunner().run(
            fan_out_fan_in, catalog, hook_manager=_create_hook_manager()
        )
        assert result_first_run == result_second_run


@pytest.mark.parametrize("is_async", [False, True])
class TestSeqentialRunnerBranchlessPipeline:
    def test_no_input_seq(self, is_async: bool, branchless_no_input_pipeline: Any, catalog: DataCatalog) -> None:
        outputs: Dict[str, Any] = SequentialRunner(is_async=is_async).run(branchless_no_input_pipeline, catalog)
        assert "E" in outputs
        assert len(outputs) == 1

    def test_no_datasets(self, is_async: bool, branchless_pipeline: Any) -> None:
        catalog = DataCatalog({}, {"ds1": 42})
        outputs: Dict[str, Any] = SequentialRunner(is_async=is_async).run(branchless_pipeline, catalog)
        assert "ds3" in outputs
        assert outputs["ds3"] == 42

    def test_no_feed(self, is_async: bool, memory_catalog: DataCatalog, branchless_pipeline: Any) -> None:
        outputs: Dict[str, Any] = SequentialRunner(is_async=is_async).run(branchless_pipeline, memory_catalog)
        assert "ds3" in outputs
        assert outputs["ds3"]["data"] == 42

    def test_node_returning_none(self, is_async: bool, saving_none_pipeline: Any, catalog: DataCatalog) -> None:
        pattern = "Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            SequentialRunner(is_async=is_async).run(saving_none_pipeline, catalog)

    def test_result_saved_not_returned(self, is_async: bool, saving_result_pipeline: Any) -> None:
        def _load() -> int:
            return 0

        def _save(arg: int) -> None:
            assert arg == 0

        catalog = DataCatalog({
            "ds": LambdaDataset(load=_load, save=_save),
            "dsX": LambdaDataset(load=_load, save=_save),
        })
        output: Dict[str, Any] = SequentialRunner(is_async=is_async).run(saving_result_pipeline, catalog)
        assert output == {}


@pytest.mark.parametrize("is_async", [False, True])
class TestSequentialRunnerBranchedPipeline:
    def test_input_seq(
        self, is_async: bool, memory_catalog: DataCatalog, unfinished_outputs_pipeline: Any, pandas_df_feed_dict: Dict[str, Any]
    ) -> None:
        memory_catalog.add_feed_dict(pandas_df_feed_dict, replace=True)
        outputs: Dict[str, Any] = SequentialRunner(is_async=is_async).run(unfinished_outputs_pipeline, memory_catalog)
        assert set(outputs.keys()) == {"ds8", "ds5", "ds6"}
        assert outputs["ds5"] == [1, 2, 3, 4, 5]
        assert isinstance(outputs["ds8"], dict)
        assert outputs["ds8"]["data"] == 42
        assert isinstance(outputs["ds6"], pd.DataFrame)

    def test_conflict_feed_catalog(
        self, is_async: bool, memory_catalog: DataCatalog, unfinished_outputs_pipeline: Any, conflicting_feed_dict: Dict[str, Any]
    ) -> None:
        memory_catalog.add_feed_dict(conflicting_feed_dict, replace=True)
        outputs: Dict[str, Any] = SequentialRunner(is_async=is_async).run(unfinished_outputs_pipeline, memory_catalog)
        assert isinstance(outputs["ds8"], dict)
        assert outputs["ds8"]["data"] == 0
        assert isinstance(outputs["ds6"], pd.DataFrame)

    def test_unsatisfied_inputs(self, is_async: bool, unfinished_outputs_pipeline: Any, catalog: DataCatalog) -> None:
        with pytest.raises(ValueError, match=f"not found in the {catalog.__class__.__name__}"):
            SequentialRunner(is_async=is_async).run(unfinished_outputs_pipeline, catalog)


class LoggingDataset(AbstractDataset):
    def __init__(self, log: List[Tuple[str, str]], name: str, value: Optional[Any] = None) -> None:
        self.log: List[Tuple[str, str]] = log
        self.name: str = name
        self.value: Optional[Any] = value

    def _load(self) -> Any:
        self.log.append(("load", self.name))
        return self.value

    def _save(self, data: Any) -> None:
        self.value = data

    def _release(self) -> None:
        self.log.append(("release", self.name))
        self.value = None

    def _describe(self) -> Dict[str, Any]:
        return {}


@pytest.mark.parametrize("is_async", [False, True])
class TestSequentialRunnerRelease:
    def test_dont_release_inputs_and_outputs(self, is_async: bool) -> None:
        log: List[Tuple[str, str]] = []
        test_pipeline: Pipeline = modular_pipeline([node(identity, "in", "middle"), node(identity, "middle", "out")])
        catalog = DataCatalog({
            "in": LoggingDataset(log, "in", "stuff"),
            "middle": LoggingDataset(log, "middle"),
            "out": LoggingDataset(log, "out"),
        })
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)
        assert log == [("load", "in"), ("load", "middle"), ("release", "middle")]

    def test_release_at_earliest_opportunity(self, is_async: bool) -> None:
        log: List[Tuple[str, str]] = []
        test_pipeline: Pipeline = modular_pipeline([
            node(source, None, "first"),
            node(identity, "first", "second"),
            node(sink, "second", None),
        ])
        catalog = DataCatalog({
            "first": LoggingDataset(log, "first"),
            "second": LoggingDataset(log, "second"),
        })
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)
        assert log == [("load", "first"), ("release", "first"), ("load", "second"), ("release", "second")]

    def test_count_multiple_loads(self, is_async: bool) -> None:
        log: List[Tuple[str, str]] = []
        test_pipeline: Pipeline = modular_pipeline([
            node(source, None, "dataset"),
            node(sink, "dataset", None, name="bob"),
            node(sink, "dataset", None, name="fred"),
        ])
        catalog = DataCatalog({"dataset": LoggingDataset(log, "dataset")})
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)
        assert log == [("load", "dataset"), ("load", "dataset"), ("release", "dataset")]

    def test_release_transcoded(self, is_async: bool) -> None:
        log: List[Tuple[str, str]] = []
        test_pipeline: Pipeline = modular_pipeline([
            node(source, None, "ds@save"),
            node(sink, "ds@load", None),
        ])
        catalog = DataCatalog({
            "ds@save": LoggingDataset(log, "save"),
            "ds@load": LoggingDataset(log, "load"),
        })
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)
        assert log == [("release", "save"), ("load", "load"), ("release", "load")]

    @pytest.mark.parametrize(
        "test_pipeline",
        [
            modular_pipeline([node(identity, "ds1", "ds2", confirms="ds1")]),
            modular_pipeline([node(identity, "ds1", "ds2"), node(identity, "ds2", None, confirms="ds1")]),
        ],
    )
    def test_confirms(self, mocker: Any, test_pipeline: Pipeline, is_async: bool) -> None:
        fake_dataset_instance: Any = mocker.Mock()
        catalog = DataCatalog(datasets={"ds1": fake_dataset_instance})
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)
        fake_dataset_instance.confirm.assert_called_once_with()


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
        caplog: Any,
        two_branches_crossed_pipeline: Pipeline,
        persistent_dataset_catalog: DataCatalog,
        failing_node_names: List[str],
        expected_pattern: str,
    ) -> None:
        nodes: Dict[str, Any] = {n.name: n for n in two_branches_crossed_pipeline.nodes}
        for name in failing_node_names:
            two_branches_crossed_pipeline -= modular_pipeline([nodes[name]])
            two_branches_crossed_pipeline += modular_pipeline([nodes[name]._copy(func=exception_fn)])
        with pytest.raises(Exception):
            SequentialRunner().run(two_branches_crossed_pipeline, persistent_dataset_catalog, hook_manager=_create_hook_manager())
        assert re.search(expected_pattern, caplog.text)

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
        caplog: Any,
        two_branches_crossed_pipeline_variable_inputs: Pipeline,
        persistent_dataset_catalog: DataCatalog,
        failing_node_names: List[str],
        expected_pattern: str,
    ) -> None:
        test_pipeline: Pipeline = two_branches_crossed_pipeline_variable_inputs
        nodes: Dict[str, Any] = {n.name: n for n in test_pipeline.nodes}
        for name in failing_node_names:
            test_pipeline -= modular_pipeline([nodes[name]])
            test_pipeline += modular_pipeline([nodes[name]._copy(func=exception_fn)])
        with pytest.raises(Exception, match="test exception"):
            SequentialRunner().run(test_pipeline, persistent_dataset_catalog, hook_manager=_create_hook_manager())
        assert re.search(expected_pattern, caplog.text)


class TestMemoryDatasetBehaviour:
    def test_run_includes_memory_datasets(self, pipeline_with_memory_datasets: Pipeline) -> None:
        catalog = DataCatalog({
            "Input1": LambdaDataset(load=lambda: "data1", save=lambda data: None),
            "Input2": LambdaDataset(load=lambda: "data2", save=lambda data: None),
            "MemOutput1": MemoryDataset(),
            "MemOutput2": MemoryDataset(),
        })
        catalog.add("RegularOutput", LambdaDataset(None, None, lambda: True))
        output: Dict[str, Any] = SequentialRunner().run(pipeline_with_memory_datasets, catalog)
        assert "MemOutput1" in output
        assert "MemOutput2" in output
        assert "RegularOutput" not in output