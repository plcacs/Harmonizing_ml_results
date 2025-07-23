from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional, Union, cast
from unittest.mock import call, Mock, patch
from uuid import UUID, uuid4
import pytest
from flask import current_app
from flask.ctx import AppContext
from flask_appbuilder.security.sqla.models import User
from flask_sqlalchemy import BaseQuery
from freezegun import freeze_time
from slack_sdk.errors import (
    BotUserAccessError,
    SlackApiError,
    SlackClientConfigurationError,
    SlackClientError,
    SlackClientNotConnectedError,
    SlackObjectFormationError,
    SlackRequestError,
    SlackTokenRotationError,
)
from sqlalchemy.sql import func
from superset import db
from superset.commands.report.exceptions import (
    AlertQueryError,
    AlertQueryInvalidTypeError,
    AlertQueryMultipleColumnsError,
    AlertQueryMultipleRowsError,
    ReportScheduleClientErrorsException,
    ReportScheduleCsvFailedError,
    ReportScheduleCsvTimeout,
    ReportScheduleNotFoundError,
    ReportSchedulePreviousWorkingError,
    ReportScheduleScreenshotFailedError,
    ReportScheduleScreenshotTimeout,
    ReportScheduleSystemErrorsException,
    ReportScheduleWorkingTimeoutError,
)
from superset.commands.report.execute import (
    AsyncExecuteReportScheduleCommand,
    BaseReportState,
)
from superset.commands.report.log_prune import AsyncPruneReportScheduleLogCommand
from superset.exceptions import SupersetException
from superset.key_value.models import KeyValueEntry
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.reports.models import (
    ReportDataFormat,
    ReportExecutionLog,
    ReportRecipientType,
    ReportSchedule,
    ReportScheduleType,
    ReportScheduleValidatorType,
    ReportState,
)
from superset.reports.notifications.exceptions import (
    NotificationError,
    NotificationParamException,
)
from superset.tasks.types import ExecutorType
from superset.utils import json
from superset.utils.database import get_example_database
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,
    load_birth_names_data,
)
from tests.integration_tests.fixtures.tabbed_dashboard import tabbed_dashboard
from tests.integration_tests.fixtures.world_bank_dashboard import (
    load_world_bank_dashboard_with_slices_module_scope,
    load_world_bank_data,
)
from tests.integration_tests.reports.utils import (
    cleanup_report_schedule,
    create_report_notification,
    CSV_FILE,
    DEFAULT_OWNER_EMAIL,
    reset_key_values,
    SCREENSHOT_FILE,
    TEST_ID,
)
from tests.integration_tests.test_app import app

pytestmark = pytest.mark.usefixtures(
    "load_world_bank_dashboard_with_slices_module_scope"
)


def get_target_from_report_schedule(report_schedule: ReportSchedule) -> List[str]:
    return [
        json.loads(recipient.recipient_config_json)["target"]
        for recipient in report_schedule.recipients
    ]


def get_cctarget_from_report_schedule(report_schedule: ReportSchedule) -> List[str]:
    return [
        json.loads(recipient.recipient_config_json).get("ccTarget", "")
        for recipient in report_schedule.recipients
    ]


def get_bcctarget_from_report_schedule(report_schedule: ReportSchedule) -> List[str]:
    return [
        json.loads(recipient.recipient_config_json).get("bccTarget", "")
        for recipient in report_schedule.recipients
    ]


def get_error_logs_query(report_schedule: ReportSchedule) -> BaseQuery:
    return (
        db.session.query(ReportExecutionLog)
        .filter(
            ReportExecutionLog.report_schedule == report_schedule,
            ReportExecutionLog.state == ReportState.ERROR,
        )
        .order_by(ReportExecutionLog.end_dttm.desc())
    )


def get_notification_error_sent_count(report_schedule: ReportSchedule) -> int:
    logs = get_error_logs_query(report_schedule).all()
    notification_sent_logs = [
        log.error_message
        for log in logs
        if log.error_message == "Notification sent with error"
    ]
    return len(notification_sent_logs)


def assert_log(state: ReportState, error_message: Optional[str] = None) -> None:
    db.session.commit()
    logs = db.session.query(ReportExecutionLog).all()
    if state == ReportState.ERROR:
        assert len(logs) == 3
    else:
        assert len(logs) == 2
    log_states = [log.state for log in logs]
    assert ReportState.WORKING in log_states
    assert state in log_states
    assert error_message in [log.error_message for log in logs]
    for log in logs:
        if log.state == ReportState.WORKING:
            assert log.value is None
            assert log.value_row_json is None


@contextmanager
def create_test_table_context(database: Database) -> Generator[Any, None, None]:
    with database.get_sqla_engine() as engine:
        engine.execute(
            "CREATE TABLE IF NOT EXISTS test_table AS SELECT 1 as first, 2 as second"
        )
        engine.execute("INSERT INTO test_table (first, second) VALUES (1, 2)")
        engine.execute("INSERT INTO test_table (first, second) VALUES (3, 4)")
    yield db.session
    with database.get_sqla_engine() as engine:
        engine.execute("DROP TABLE test_table")


@pytest.fixture
def create_report_email_chart() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        email_target="target@email.com", chart=chart
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_chart_with_cc_and_bcc() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        email_target="target@email.com",
        ccTarget="cc@email.com",
        bccTarget="bcc@email.com",
        chart=chart,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_chart_alpha_owner(
    get_user: Any,
) -> Generator[ReportSchedule, None, None]:
    owners = [get_user("alpha")]
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        email_target="target@email.com", chart=chart, owners=owners
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_chart_force_screenshot() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        email_target="target@email.com", chart=chart, force_screenshot=True
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_chart_with_csv() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        email_target="target@email.com",
        chart=chart,
        report_format=ReportDataFormat.CSV,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_chart_with_text() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        email_target="target@email.com",
        chart=chart,
        report_format=ReportDataFormat.TEXT,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_chart_with_csv_no_query_context() -> Generator[
    ReportSchedule, None, None
]:
    chart = db.session.query(Slice).first()
    chart.query_context = None
    report_schedule = create_report_notification(
        email_target="target@email.com",
        chart=chart,
        report_format=ReportDataFormat.CSV,
        name="report_csv_no_query_context",
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_dashboard() -> Generator[ReportSchedule, None, None]:
    dashboard = db.session.query(Dashboard).first()
    report_schedule = create_report_notification(
        email_target="target@email.com", dashboard=dashboard
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_email_dashboard_force_screenshot() -> Generator[
    ReportSchedule, None, None
]:
    dashboard = db.session.query(Dashboard).first()
    report_schedule = create_report_notification(
        email_target="target@email.com", dashboard=dashboard, force_screenshot=True
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_slack_chart() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        slack_channel="slack_channel", chart=chart
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_slack_chartv2() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        slack_channel="slack_channel_id", chart=chart, name="report_slack_chartv2"
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_slack_chart_with_csv() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        slack_channel="slack_channel",
        chart=chart,
        report_format=ReportDataFormat.CSV,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_slack_chart_with_text() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        slack_channel="slack_channel",
        chart=chart,
        report_format=ReportDataFormat.TEXT,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_report_slack_chart_working() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        slack_channel="slack_channel", chart=chart
    )
    report_schedule.last_state = ReportState.WORKING
    report_schedule.last_eval_dttm = datetime(2020, 1, 1, 0, 0)
    report_schedule.last_value = None
    report_schedule.last_value_row_json = None
    db.session.commit()
    log = ReportExecutionLog(
        scheduled_dttm=report_schedule.last_eval_dttm,
        start_dttm=report_schedule.last_eval_dttm,
        end_dttm=report_schedule.last_eval_dttm,
        value=report_schedule.last_value,
        value_row_json=report_schedule.last_value_row_json,
        state=ReportState.WORKING,
        report_schedule=report_schedule,
        uuid=uuid4(),
    )
    db.session.add(log)
    db.session.commit()
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def create_alert_slack_chart_success() -> Generator[ReportSchedule, None, None]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        slack_channel="slack_channel",
        chart=chart,
        report_type=ReportScheduleType.ALERT,
    )
    report_schedule.last_state = ReportState.SUCCESS
    report_schedule.last_eval_dttm = datetime(2020, 1, 1, 0, 0)
    log = ReportExecutionLog(
        report_schedule=report_schedule,
        state=ReportState.SUCCESS,
        start_dttm=report_schedule.last_eval_dttm,
        end_dttm=report_schedule.last_eval_dttm,
        scheduled_dttm=report_schedule.last_eval_dttm,
    )
    db.session.add(log)
    db.session.commit()
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture(params=["alert1"])
def create_alert_slack_chart_grace(
    request: pytest.FixtureRequest,
) -> Generator[ReportSchedule, None, None]:
    param_config = {
        "alert1": {
            "sql": "SELECT count(*) from test_table",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": "<", "threshold": 10}',
        }
    }
    chart = db.session.query(Slice).first()
    example_database = get_example_database()
    with create_test_table_context(example_database):
        report_schedule = create_report_notification(
            slack_channel="slack_channel",
            chart=chart,
            report_type=ReportScheduleType.ALERT,
            database=example_database,
            sql=param_config[request.param]["sql"],
            validator_type=param_config[request.param]["validator_type"],
            validator_config_json=param_config[request.param]["validator_config_json"],
        )
        report_schedule.last_state = ReportState.GRACE
        report_schedule.last_eval_dttm = datetime(2020, 1, 1, 0, 0)
        log = ReportExecutionLog(
            report_schedule=report_schedule,
            state=ReportState.SUCCESS,
            start_dttm=report_schedule.last_eval_dttm,
            end_dttm=report_schedule.last_eval_dttm,
            scheduled_dttm=report_schedule.last_eval_dttm,
        )
        db.session.add(log)
        db.session.commit()
        yield report_schedule
        cleanup_report_schedule(report_schedule)


@pytest.fixture(
    params=[
        "alert1",
        "alert2",
        "alert3",
        "alert4",
        "alert5",
        "alert6",
        "alert7",
        "alert8",
    ]
)
def create_alert_email_chart(
    request: pytest.FixtureRequest,
) -> Generator[ReportSchedule, None, None]:
    param_config = {
        "alert1": {
            "sql": "SELECT 10 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": ">", "threshold": 9}',
        },
        "alert2": {
            "sql": "SELECT 10 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": ">=", "threshold": 10}',
        },
        "alert3": {
            "sql": "SELECT 10 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": "<", "threshold": 11}',
        },
        "alert4": {
            "sql": "SELECT 10 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": "<=", "threshold": 10}',
        },
        "alert5": {
            "sql": "SELECT 10 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": "!=", "threshold": 11}',
        },
        "alert6": {
            "sql": "SELECT 'something' as metric",
            "validator_type": ReportScheduleValidatorType.NOT_NULL,
            "validator_config_json": "{}",
        },
        "alert7": {
            "sql": "SELECT {{ 5 + 5 }} as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": "!=", "threshold": 11}',
        },
        "alert8": {
            "sql": "SELECT 55 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": ">", "threshold": 54.999}',
        },
    }
    chart = db.session.query(Slice).first()
    example_database = get_example_database()
    with create_test_table_context(example_database):
        report_schedule = create_report_notification(
            email_target="target@email.com",
            chart=chart,
            report_type=ReportScheduleType.ALERT,
            database=example_database,
            sql=param_config[request.param]["sql"],
            validator_type=param_config[request.param]["validator_type"],
            validator_config_json=param_config[request.param]["validator_config_json"],
            force_screenshot=True,
        )
        yield report_schedule
        cleanup_report_schedule(report_schedule)


@pytest.fixture(
    params=[
        "alert1",
        "alert2",
        "alert3",
        "alert4",
        "alert5",
        "alert6",
        "alert7",
        "alert8",
        "alert9",
    ]
)
def create_no_alert_email_chart(
    request: pytest.FixtureRequest,
) -> Generator[ReportSchedule, None, None]:
    param_config = {
        "alert1": {
            "sql": "SELECT 10 as metric",
            "validator_type": ReportScheduleValidatorType.OPERATOR,
            "validator_config_json": '{"op": "<", "threshold": 10}',
        },
        "alert2": {
            "sql": "SELECT 10 as