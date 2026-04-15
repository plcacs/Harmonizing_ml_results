import json
from typing import Any, Callable, Dict, List, Optional, Union
from unittest import mock
import pytest
from pytest_mock import MockerFixture
from dbt.events.types import (
    ArtifactWritten,
    EndOfRunSummary,
    GenericExceptionOnRun,
    InvalidConcurrentBatchesConfig,
    JinjaLogDebug,
    LogBatchResult,
    LogModelResult,
    MicrobatchExecutionDebug,
    MicrobatchMacroOutsideOfBatchesDeprecation,
    MicrobatchModelNoEventTimeInputs,
)
from dbt.tests.fixtures.project import TestProjInfo
from dbt.tests.util import (
    get_artifact,
    patch_microbatch_end_time,
    read_file,
    relation_from_name,
    run_dbt,
    run_dbt_and_capture,
    write_file,
)
from tests.utils import EventCatcher

input_model_sql: str = ...
input_model_invalid_sql: str = ...
input_model_without_event_time_sql: str = ...
microbatch_model_sql: str = ...
microbatch_model_with_pre_and_post_sql: str = ...
microbatch_model_force_concurrent_batches_sql: str = ...
microbatch_yearly_model_sql: str = ...
microbatch_yearly_model_downstream_sql: str = ...
invalid_batch_jinja_context_macro_sql: str = ...
microbatch_model_with_context_checks_sql: str = ...
microbatch_model_downstream_sql: str = ...
microbatch_model_ref_render_sql: str = ...
seed_csv: str = ...
seeds_yaml: str = ...
sources_yaml: str = ...
microbatch_model_calling_source_sql: str = ...
custom_microbatch_strategy: str = ...
downstream_model_of_microbatch_sql: str = ...
microbatch_model_full_refresh_false_sql: str = ...
microbatch_model_context_vars: str = ...
microbatch_model_failing_incremental_partition_sql: str = ...
microbatch_model_first_partition_failing_sql: str = ...
microbatch_model_second_batch_failing_sql: str = ...


class BaseMicrobatchCustomUserStrategy:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]: ...

    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Dict[str, bool]]: ...

    @pytest.fixture(scope="class")
    def deprecation_catcher(self) -> EventCatcher: ...


class TestMicrobatchCustomUserStrategyDefault(BaseMicrobatchCustomUserStrategy):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Dict[str, bool]]: ...

    def test_use_custom_microbatch_strategy_by_default(
        self, project: TestProjInfo, deprecation_catcher: EventCatcher
    ) -> None: ...


class TestMicrobatchCustomUserStrategyProjectFlagTrueValid(
    BaseMicrobatchCustomUserStrategy
):
    def test_use_custom_microbatch_strategy_project_flag_true_invalid_incremental_strategy(
        self, project: TestProjInfo, deprecation_catcher: EventCatcher
    ) -> None: ...


class TestMicrobatchCustomUserStrategyProjectFlagTrueNoValidBuiltin(
    BaseMicrobatchCustomUserStrategy
):
    def test_use_custom_microbatch_strategy_project_flag_true_invalid_incremental_strategy(
        self, project: TestProjInfo
    ) -> None: ...


class BaseMicrobatchTest:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def assert_row_count(
        self, project: TestProjInfo, relation_name: str, expected_row_count: int
    ) -> None: ...


class TestMicrobatchCLI(BaseMicrobatchTest):
    CLI_COMMAND_NAME: str = ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchCLIBuild(TestMicrobatchCLI):
    CLI_COMMAND_NAME: str = ...


class TestMicrobatchCLIRunOutputJSON(BaseMicrobatchTest):
    def test_list_output_json(self, project: TestProjInfo) -> None: ...


class TestMicroBatchBoundsDefault(BaseMicrobatchTest):
    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchWithSource(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def seeds(self) -> Dict[str, str]: ...

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchJinjaContext(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]: ...

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchWithInputWithoutEventTime(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchUsingRefRenderSkipsFilter(BaseMicrobatchTest):
    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchJinjaContextVarsAvailable(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time_logs(self, project: TestProjInfo) -> None: ...


class TestMicrobatchIncrementalBatchFailure(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchRetriesPartialSuccesses(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchMultipleRetries(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchInitialBatchFailure(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchSecondBatchFailure(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchCompiledRunPaths(BaseMicrobatchTest):
    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrobatchFullRefreshConfigFalse(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_run_with_event_time(self, project: TestProjInfo) -> None: ...


class TestMicrbobatchModelsRunWithSameCurrentTime(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_microbatch(self, project: TestProjInfo) -> None: ...


class TestMicrobatchModelStoppedByKeyboardInterrupt(BaseMicrobatchTest):
    @pytest.fixture
    def catch_eors(self) -> EventCatcher: ...

    @pytest.fixture
    def catch_aw(self) -> EventCatcher: ...

    def test_microbatch(
        self,
        mocker: MockerFixture,
        project: TestProjInfo,
        catch_eors: EventCatcher,
        catch_aw: EventCatcher,
    ) -> None: ...


class TestMicrobatchModelSkipped(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    def test_microbatch_model_skipped(self, project: TestProjInfo) -> None: ...


class TestMicrobatchCanRunParallelOrSequential(BaseMicrobatchTest):
    @pytest.fixture
    def batch_exc_catcher(self) -> EventCatcher: ...

    def test_microbatch(
        self,
        mocker: MockerFixture,
        project: TestProjInfo,
        batch_exc_catcher: EventCatcher,
    ) -> None: ...


class TestFirstAndLastBatchAlwaysSequential(BaseMicrobatchTest):
    @pytest.fixture
    def batch_exc_catcher(self) -> EventCatcher: ...

    def test_microbatch(
        self,
        mocker: MockerFixture,
        project: TestProjInfo,
        batch_exc_catcher: EventCatcher,
    ) -> None: ...


class TestFirstBatchRunsPreHookLastBatchRunsPostHook(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    @pytest.fixture
    def batch_log_catcher(self) -> EventCatcher: ...

    def test_microbatch(
        self,
        mocker: MockerFixture,
        project: TestProjInfo,
        batch_log_catcher: EventCatcher,
    ) -> None: ...


class TestWhenOnlyOneBatchRunBothPostAndPreHooks(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    @pytest.fixture
    def batch_log_catcher(self) -> EventCatcher: ...

    @pytest.fixture
    def generic_exception_catcher(self) -> EventCatcher: ...

    def test_microbatch(
        self,
        project: TestProjInfo,
        batch_log_catcher: EventCatcher,
        generic_exception_catcher: EventCatcher,
    ) -> None: ...


class TestCanSilenceInvalidConcurrentBatchesConfigWarning(BaseMicrobatchTest):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...

    @pytest.fixture
    def event_catcher(self) -> EventCatcher: ...

    def test_microbatch(
        self, project: TestProjInfo, event_catcher: EventCatcher
    ) -> None: ...