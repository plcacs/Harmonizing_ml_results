from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Type
import pytest
from kedro.framework.hooks import _create_hook_manager
from kedro.io import (
    AbstractDataset,
    DataCatalog,
    DatasetError,
    KedroDataCatalog,
    MemoryDataset,
)
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline, pipeline as modular_pipeline
from kedro.runner import ThreadRunner
from tests.runner.conftest import exception_fn, identity, return_none, sink, source


class TestValidThreadRunner:

    def test_thread_run(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None:
        catalog.add_feed_dict({'A': 42})
        result: dict[str, Any] = ThreadRunner().run(fan_out_fan_in, catalog)
        assert 'Z' in result
        assert result['Z'] == (42, 42, 42)

    def test_thread_run_with_plugin_manager(
        self, fan_out_fan_in: Pipeline, catalog: DataCatalog
    ) -> None:
        catalog.add_feed_dict({'A': 42})
        result: dict[str, Any] = ThreadRunner().run(
            fan_out_fan_in, catalog, hook_manager=_create_hook_manager()
        )
        assert 'Z' in result
        assert result['Z'] == (42, 42, 42)

    def test_memory_dataset_input(self, fan_out_fan_in: Pipeline) -> None:
        catalog: DataCatalog = DataCatalog({'A': MemoryDataset('42')})
        result: dict[str, Any] = ThreadRunner().run(fan_out_fan_in, catalog)
        assert 'Z' in result
        assert result['Z'] == ('42', '42', '42')

    def test_does_not_log_not_using_async(
        self, fan_out_fan_in: Pipeline, catalog: DataCatalog, caplog: Any
    ) -> None:
        catalog.add_feed_dict({'A': 42})
        ThreadRunner().run(fan_out_fan_in, catalog)
        assert 'Using synchronous mode for loading and saving data.' not in caplog.text

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_thread_run_with_patterns(
        self, catalog_type: Type[DataCatalog]
    ) -> None:
        """Test warm-up is done and patterns are resolved before running pipeline.

        Without the warm-up "Dataset 'dummy_1' has already been registered" error
        would be raised for this test. We check that the dataset was registered at the
        warm-up, and we successfully passed to loading it.
        """
        catalog_conf: dict[str, Any] = {'{catch_all}': {'type': 'MemoryDataset'}}
        catalog: DataCatalog = catalog_type.from_config(catalog_conf)
        test_pipeline: Pipeline = pipeline([
            node(identity, inputs='dummy_1', outputs='output_1', name='node_1'),
            node(identity, inputs='dummy_2', outputs='output_2', name='node_2'),
            node(identity, inputs='dummy_1', outputs='output_3', name='node_3'),
        ])
        with pytest.raises(Exception, match='Data for MemoryDataset has not been saved yet'):
            ThreadRunner().run(test_pipeline, catalog)


class TestMaxWorkers:

    @pytest.mark.parametrize(
        'user_specified_number, expected_number',
        [(6, 3), (None, 3)]
    )
    def test_specified_max_workers(
        self,
        mocker: Any,
        fan_out_fan_in: Pipeline,
        catalog: DataCatalog,
        user_specified_number: int | None,
        expected_number: int
    ) -> None:
        """
        We initialize the runner with max_workers=4.
        `fan_out_fan_in` pipeline needs 3 threads.
        A pool with 3 workers should be used.
        """
        executor_cls_mock: Any = mocker.patch(
            'kedro.runner.thread_runner.ThreadPoolExecutor',
            wraps=ThreadPoolExecutor
        )
        catalog.add_feed_dict({'A': 42})
        result: dict[str, Any] = ThreadRunner(max_workers=user_specified_number).run(
            fan_out_fan_in, catalog
        )
        assert result == {'Z': (42, 42, 42)}
        executor_cls_mock.assert_called_once_with(max_workers=expected_number)

    def test_init_with_negative_process_count(self) -> None:
        pattern: str = 'max_workers should be positive'
        with pytest.raises(ValueError, match=pattern):
            ThreadRunner(max_workers=-1)


class TestIsAsync:

    def test_thread_run(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None:
        catalog.add_feed_dict({'A': 42})
        pattern: str = (
            "'ThreadRunner' doesn't support loading and saving the node inputs and outputs "
            "asynchronously with threads. Setting 'is_async' to False."
        )
        with pytest.warns(UserWarning, match=pattern):
            result: dict[str, Any] = ThreadRunner(is_async=True).run(
                fan_out_fan_in, catalog
            )
        assert 'Z' in result
        assert result['Z'] == (42, 42, 42)


class TestInvalidThreadRunner:

    def test_task_exception(self, fan_out_fan_in: Pipeline, catalog: DataCatalog) -> None:
        catalog.add_feed_dict(feed_dict={'A': 42})
        pipeline: Pipeline = modular_pipeline([
            fan_out_fan_in,
            node(exception_fn, 'Z', 'X')
        ])
        with pytest.raises(Exception, match='test exception'):
            ThreadRunner().run(pipeline, catalog)

    def test_node_returning_none(self) -> None:
        pipeline: Pipeline = modular_pipeline([
            node(identity, 'A', 'B'),
            node(return_none, 'B', 'C'),
        ])
        catalog: DataCatalog = DataCatalog({'A': MemoryDataset('42')})
        pattern: str = "Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            ThreadRunner().run(pipeline, catalog)


class LoggingDataset(AbstractDataset):

    def __init__(self, log: list[tuple[str, str]], name: str, value: Any = None) -> None:
        self.log: list[tuple[str, str]] = log
        self.name: str = name
        self.value: Any = value

    def _load(self) -> Any:
        self.log.append(('load', self.name))
        return self.value

    def _save(self, data: Any) -> None:
        self.value = data

    def _release(self) -> None:
        self.log.append(('release', self.name))
        self.value = None

    def _describe(self) -> dict[str, Any]:
        return {}


class TestThreadRunnerRelease:

    def test_dont_release_inputs_and_outputs(self) -> None:
        log: list[tuple[str, str]] = []
        pipeline: Pipeline = modular_pipeline([
            node(identity, 'in', 'middle'),
            node(identity, 'middle', 'out'),
        ])
        catalog: DataCatalog = DataCatalog({
            'in': LoggingDataset(log, 'in', 'stuff'),
            'middle': LoggingDataset(log, 'middle'),
            'out': LoggingDataset(log, 'out'),
        })
        ThreadRunner().run(pipeline, catalog)
        assert list(log) == [
            ('load', 'in'),
            ('load', 'middle'),
            ('release', 'middle'),
        ]

    def test_release_at_earliest_opportunity(self) -> None:
        runner: ThreadRunner = ThreadRunner()
        log: list[tuple[str, str]] = []
        pipeline: Pipeline = modular_pipeline([
            node(source, None, 'first'),
            node(identity, 'first', 'second'),
            node(sink, 'second', None),
        ])
        catalog: DataCatalog = DataCatalog({
            'first': LoggingDataset(log, 'first'),
            'second': LoggingDataset(log, 'second'),
        })
        runner.run(pipeline, catalog)
        assert list(log) == [
            ('load', 'first'),
            ('release', 'first'),
            ('load', 'second'),
            ('release', 'second'),
        ]

    def test_count_multiple_loads(self) -> None:
        runner: ThreadRunner = ThreadRunner()
        log: list[tuple[str, str]] = []
        pipeline: Pipeline = modular_pipeline([
            node(source, None, 'dataset'),
            node(sink, 'dataset', None, name='bob'),
            node(sink, 'dataset', None, name='fred'),
        ])
        catalog: DataCatalog = DataCatalog({
            'dataset': LoggingDataset(log, 'dataset'),
        })
        runner.run(pipeline, catalog)
        assert list(log) == [
            ('load', 'dataset'),
            ('load', 'dataset'),
            ('release', 'dataset'),
        ]

    def test_release_transcoded(self) -> None:
        log: list[tuple[str, str]] = []
        pipeline: Pipeline = modular_pipeline([
            node(source, None, 'ds@save'),
            node(sink, 'ds@load', None),
        ])
        catalog: DataCatalog = DataCatalog({
            'ds@save': LoggingDataset(log, 'save'),
            'ds@load': LoggingDataset(log, 'load'),
        })
        ThreadRunner().run(pipeline, catalog)
        assert list(log) == [
            ('release', 'save'),
            ('load', 'load'),
            ('release', 'load'),
        ]


class TestSuggestResumeScenario:

    @pytest.mark.parametrize(
        'failing_node_names,expected_pattern',
        [
            (['node1_A', 'node1_B'], 'No nodes ran.'),
            (['node2'], '(node1_A,node1_B|node1_B,node1_A)'),
            (['node3_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'),
            (['node4_A'], '(node3_A,node3_B|node3_B,node3_A|node3_A)'),
            (
                ['node3_A', 'node4_A'],
                '(node3_A,node3_B|node3_B,node3_A|node3_A)'
            ),
            (
                ['node2', 'node4_A'],
                '(node1_A,node1_B|node1_B,node1_A)'
            ),
        ]
    )
    def test_suggest_resume_scenario(
        self,
        caplog: Any,
        two_branches_crossed_pipeline: Pipeline,
        persistent_dataset_catalog: DataCatalog,
        failing_node_names: list[str],
        expected_pattern: str
    ) -> None:
        nodes: dict[str, node] = {n.name: n for n in two_branches_crossed_pipeline.nodes}
        for name in failing_node_names:
            two_branches_crossed_pipeline -= modular_pipeline([nodes[name]])
            two_branches_crossed_pipeline += modular_pipeline([
                nodes[name]._copy(func=exception_fn)
            ])
        with pytest.raises(Exception):
            ThreadRunner().run(
                two_branches_crossed_pipeline,
                persistent_dataset_catalog,
                hook_manager=_create_hook_manager(),
            )
        assert re.search(expected_pattern, caplog.text)

    @pytest.mark.parametrize(
        'failing_node_names,expected_pattern',
        [
            (['node1_A', 'node1_B'], 'No nodes ran.'),
            (['node2'], '"node1_A,node1_B"'),
            (['node3_A'], '(node3_A,node3_B|node3_A)'),
            (['node4_A'], '(node3_A,node3_B|node3_A)'),
            (
                ['node3_A', 'node4_A'],
                '(node3_A,node3_B|node3_A)'
            ),
            (
                ['node2', 'node4_A'],
                '"node1_A,node1_B"'
            ),
        ]
    )
    def test_stricter_suggest_resume_scenario(
        self,
        caplog: Any,
        two_branches_crossed_pipeline_variable_inputs: Pipeline,
        persistent_dataset_catalog: DataCatalog,
        failing_node_names: list[str],
        expected_pattern: str
    ) -> None:
        """
        Stricter version of previous test.
        Covers pipelines where inputs are shared across nodes.
        """
        test_pipeline: Pipeline = two_branches_crossed_pipeline_variable_inputs
        nodes: dict[str, node] = {n.name: n for n in test_pipeline.nodes}
        for name in failing_node_names:
            test_pipeline -= modular_pipeline([nodes[name]])
            test_pipeline += modular_pipeline([
                nodes[name]._copy(func=exception_fn)
            ])
        with pytest.raises(Exception, match='test exception'):
            ThreadRunner(max_workers=1).run(
                test_pipeline,
                persistent_dataset_catalog,
                hook_manager=_create_hook_manager(),
            )
        assert re.search(expected_pattern, caplog.text)
