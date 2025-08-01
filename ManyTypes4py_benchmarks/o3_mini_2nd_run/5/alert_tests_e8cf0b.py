#!/usr/bin/env python3
import uuid
from contextlib import nullcontext, suppress
from typing import Any, Callable, List, Optional, Union

import pandas as pd
import pytest
from flask.ctx import AppContext  # type: ignore
from pytest_mock import MockerFixture
from superset.commands.report.exceptions import AlertQueryError
from superset.reports.models import ReportCreationMethod, ReportScheduleType
from superset.tasks.types import ExecutorType, FixedExecutor
from superset.utils.database import get_example_database
from tests.integration_tests.test_app import app


@pytest.mark.parametrize(
    "owner_names, creator_name, config, expected_result",
    [
        (["gamma"], None, [FixedExecutor("admin")], "admin"),
        (["gamma"], None, [ExecutorType.OWNER], "gamma"),
        (["alpha", "gamma"], "gamma", [ExecutorType.CREATOR_OWNER], "gamma"),
        (["alpha", "gamma"], "alpha", [ExecutorType.CREATOR_OWNER], "alpha"),
        (["alpha", "gamma"], "admin", [ExecutorType.CREATOR_OWNER], AlertQueryError()),
        (["gamma"], None, [ExecutorType.CURRENT_USER], AlertQueryError()),
    ],
)
def test_execute_query_as_report_executor(
    owner_names: List[str],
    creator_name: Optional[str],
    config: List[Union[FixedExecutor, ExecutorType]],
    expected_result: Union[str, Exception],
    mocker: MockerFixture,
    app_context: AppContext,
    get_user: Callable[[str], Any],
) -> None:
    from superset.commands.report.alert import AlertCommand
    from superset.reports.models import ReportSchedule

    original_config: Any = app.config["ALERT_REPORTS_EXECUTORS"]
    app.config["ALERT_REPORTS_EXECUTORS"] = config
    owners: List[Any] = [get_user(owner_name) for owner_name in owner_names]
    report_schedule: Any = ReportSchedule(
        created_by=get_user(creator_name) if creator_name else None,
        owners=owners,
        type=ReportScheduleType.ALERT,
        description="description",
        crontab="0 9 * * *",
        creation_method=ReportCreationMethod.ALERTS_REPORTS,
        sql="SELECT 1",
        grace_period=14400,
        working_timeout=3600,
        database=get_example_database(),
        validator_config_json='{"op": "==", "threshold": 1}',
    )
    command: Any = AlertCommand(report_schedule=report_schedule, execution_id=uuid.uuid4())
    override_user_mock = mocker.patch("superset.commands.report.alert.override_user")
    cm = pytest.raises(type(expected_result)) if isinstance(expected_result, Exception) else nullcontext()
    with cm:
        command.run()
        assert override_user_mock.call_args[0][0].username == expected_result  # type: ignore
    app.config["ALERT_REPORTS_EXECUTORS"] = original_config


def test_execute_query_mutate_query_enabled(
    mocker: MockerFixture, app_context: AppContext, get_user: Callable[[str], Any]
) -> None:
    from superset.commands.report.alert import AlertCommand
    from superset.reports.models import ReportSchedule

    default_alert_mutate_ff: Any = app.config["MUTATE_ALERT_QUERY"]
    app.config["MUTATE_ALERT_QUERY"] = True
    mocker.patch("superset.commands.report.alert.override_user")
    mock_df: pd.DataFrame = mocker.MagicMock(spec=pd.DataFrame)
    mock_df.empty = True
    mock_database: Any = get_example_database()
    mock_get_df = mocker.patch.object(mock_database, "get_df", return_value=mock_df)
    mock_limited_sql = mocker.patch.object(mock_database, "apply_limit_to_sql")
    mock_mutate_call = mocker.patch.object(mock_database, "mutate_sql_based_on_config")
    report_schedule: Any = ReportSchedule(
        created_by=get_user("admin"),
        owners=[get_user("admin")],
        type=ReportScheduleType.ALERT,
        description="description",
        crontab="0 9 * * *",
        creation_method=ReportCreationMethod.ALERTS_REPORTS,
        sql="SELECT 1",
        grace_period=14400,
        working_timeout=3600,
        database=mock_database,
        validator_config_json='{"op": "==", "threshold": 1}',
    )
    AlertCommand(report_schedule=report_schedule, execution_id=uuid.uuid4()).run()
    mock_mutate_call.assert_called_once_with(mock_limited_sql.return_value)
    mock_get_df.assert_called_once_with(sql=mock_mutate_call.return_value)
    app.config["MUTATE_ALERT_QUERY"] = default_alert_mutate_ff


def test_execute_query_mutate_query_disabled(
    mocker: MockerFixture, app_context: AppContext, get_user: Callable[[str], Any]
) -> None:
    from superset.commands.report.alert import AlertCommand
    from superset.reports.models import ReportSchedule

    default_alert_mutate_ff: Any = app.config["MUTATE_ALERT_QUERY"]
    app.config["MUTATE_ALERT_QUERY"] = False
    mocker.patch("superset.commands.report.alert.override_user")
    mock_database: Any = mocker.MagicMock()
    report_schedule: Any = ReportSchedule(
        created_by=get_user("admin"),
        owners=[get_user("admin")],
        type=ReportScheduleType.ALERT,
        description="description",
        crontab="0 9 * * *",
        creation_method=ReportCreationMethod.ALERTS_REPORTS,
        sql="SELECT 1",
        grace_period=14400,
        working_timeout=3600,
        database=mock_database,
        validator_config_json='{"op": "==", "threshold": 1}',
    )
    AlertCommand(report_schedule=report_schedule, execution_id=uuid.uuid4()).run()
    mock_database.mutate_sql_based_on_config.assert_not_called()
    mock_database.get_df.assert_called_once_with(
        sql=mock_database.apply_limit_to_sql.return_value
    )
    app.config["MUTATE_ALERT_QUERY"] = default_alert_mutate_ff


def test_execute_query_succeeded_no_retry(
    mocker: MockerFixture, app_context: AppContext
) -> None:
    from superset.commands.report.alert import AlertCommand

    execute_query_mock = mocker.patch(
        "superset.commands.report.alert.AlertCommand._execute_query",
        side_effect=lambda: pd.DataFrame([{"sample_col": 0}]),
    )
    command: Any = AlertCommand(report_schedule=mocker.Mock(), execution_id=uuid.uuid4())
    command.validate()
    assert execute_query_mock.call_count == 1


def test_execute_query_succeeded_with_retries(
    mocker: MockerFixture, app_context: AppContext
) -> None:
    from superset.commands.report.alert import AlertCommand, AlertQueryError

    execute_query_mock = mocker.patch("superset.commands.report.alert.AlertCommand._execute_query")
    query_executed_count: int = 0
    expected_max_retries: int = 3

    def _mocked_execute_query() -> pd.DataFrame:
        nonlocal query_executed_count
        query_executed_count += 1
        if query_executed_count < expected_max_retries:
            raise AlertQueryError()
        else:
            return pd.DataFrame([{"sample_col": 0}])

    execute_query_mock.side_effect = _mocked_execute_query
    execute_query_mock.__name__ = "mocked_execute_query"  # type: ignore
    command: Any = AlertCommand(report_schedule=mocker.Mock(), execution_id=uuid.uuid4())
    command.validate()
    assert execute_query_mock.call_count == expected_max_retries


def test_execute_query_failed_no_retry(
    mocker: MockerFixture, app_context: AppContext
) -> None:
    from superset.commands.report.alert import AlertCommand, AlertQueryTimeout

    execute_query_mock = mocker.patch("superset.commands.report.alert.AlertCommand._execute_query")

    def _mocked_execute_query() -> None:
        raise AlertQueryTimeout

    execute_query_mock.side_effect = _mocked_execute_query
    execute_query_mock.__name__ = "mocked_execute_query"  # type: ignore
    command: Any = AlertCommand(report_schedule=mocker.Mock(), execution_id=uuid.uuid4())
    with suppress(AlertQueryTimeout):
        command.validate()
    assert execute_query_mock.call_count == 1


def test_execute_query_failed_max_retries(
    mocker: MockerFixture, app_context: AppContext
) -> None:
    from superset.commands.report.alert import AlertCommand, AlertQueryError

    execute_query_mock = mocker.patch("superset.commands.report.alert.AlertCommand._execute_query")

    def _mocked_execute_query() -> None:
        raise AlertQueryError

    execute_query_mock.side_effect = _mocked_execute_query
    execute_query_mock.__name__ = "mocked_execute_query"  # type: ignore
    command: Any = AlertCommand(report_schedule=mocker.Mock(), execution_id=uuid.uuid4())
    with suppress(AlertQueryError):
        command.validate()
    assert execute_query_mock.call_count == 3


def test_get_alert_metadata_from_object(
    mocker: MockerFixture, app_context: AppContext, get_user: Callable[[str], Any]
) -> None:
    from superset.commands.report.alert import AlertCommand
    from superset.reports.models import ReportSchedule

    app.config["ALERT_REPORTS_EXECUTORS"] = [ExecutorType.OWNER]
    mock_database: Any = mocker.MagicMock()
    mock_exec_id: uuid.UUID = uuid.uuid4()
    report_schedule: Any = ReportSchedule(
        created_by=get_user("admin"),
        owners=[get_user("admin")],
        type=ReportScheduleType.ALERT,
        description="description",
        crontab="0 9 * * *",
        creation_method=ReportCreationMethod.ALERTS_REPORTS,
        sql="SELECT 1",
        grace_period=14400,
        working_timeout=3600,
        database=mock_database,
        validator_config_json='{"op": "==", "threshold": 1}',
    )
    cm: Any = AlertCommand(report_schedule=report_schedule, execution_id=mock_exec_id)
    assert cm._get_alert_metadata_from_object() == {
        "report_schedule_id": report_schedule.id,
        "execution_id": mock_exec_id,
    }