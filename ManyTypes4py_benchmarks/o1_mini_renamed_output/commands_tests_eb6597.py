from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Iterator, Callable, Any
from unittest.mock import call, Mock, patch
from uuid import uuid4
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
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
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
from superset.commands.report.execute import AsyncExecuteReportScheduleCommand, BaseReportState
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
from superset.reports.notifications.exceptions import NotificationError, NotificationParamException
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

pytestmark = pytest.mark.usefixtures('load_world_bank_dashboard_with_slices_module_scope')


def func_u0jt1phh(report_schedule: ReportSchedule) -> List[str]:
    return [
        json.loads(recipient.recipient_config_json)['target']
        for recipient in report_schedule.recipients
    ]


def func_uukk02sx(report_schedule: ReportSchedule) -> List[str]:
    return [
        json.loads(recipient.recipient_config_json).get('ccTarget', '')
        for recipient in report_schedule.recipients
    ]


def func_sn4ax80g(report_schedule: ReportSchedule) -> List[str]:
    return [
        json.loads(recipient.recipient_config_json).get('bccTarget', '')
        for recipient in report_schedule.recipients
    ]


def func_mc2fu2ha(report_schedule: ReportSchedule) -> BaseQuery:
    return (
        db.session.query(ReportExecutionLog)
        .filter(
            ReportExecutionLog.report_schedule == report_schedule,
            ReportExecutionLog.state == ReportState.ERROR,
        )
        .order_by(ReportExecutionLog.end_dttm.desc())
    )


def func_xbwiqmgs(report_schedule: ReportSchedule) -> int:
    logs = func_mc2fu2ha(report_schedule).all()
    notification_sent_logs = [
        log.error_message for log in logs if log.error_message == 'Notification sent with error'
    ]
    return len(notification_sent_logs)


def func_3aof9eks(state: ReportState, error_message: Optional[str] = None) -> None:
    db.session.commit()
    logs: List[ReportExecutionLog] = db.session.query(ReportExecutionLog).all()
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
def func_9aesecuq(database: Database) -> Iterator[Session]:
    with database.get_sqla_engine() as engine:
        engine.execute(
            'CREATE TABLE IF NOT EXISTS test_table AS SELECT 1 as first, 2 as second'
        )
        engine.execute('INSERT INTO test_table (first, second) VALUES (1, 2)')
        engine.execute('INSERT INTO test_table (first, second) VALUES (3, 4)')
    yield db.session
    with database.get_sqla_engine() as engine:
        engine.execute('DROP TABLE test_table')


@pytest.fixture
def func_g2goqwrf() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(email_target='target@email.com', chart=chart)
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_ch45v7td() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        email_target='target@email.com',
        ccTarget='cc@email.com',
        bccTarget='bcc@email.com',
        chart=chart,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_c5ogbrc5(get_user: Callable[[str], User]) -> Iterator[ReportSchedule]:
    owners = [get_user('alpha')]
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(email_target='target@email.com', chart=chart, owners=owners)
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_5wzjkmzl() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(email_target='target@email.com', chart=chart, force_screenshot=True)
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_1p5tvtra() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        email_target='target@email.com',
        chart=chart,
        report_format=ReportDataFormat.CSV,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_ip02qth6() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        email_target='target@email.com',
        chart=chart,
        report_format=ReportDataFormat.TEXT,
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_th8in3uf() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    chart.query_context = None
    report_schedule = create_report_notification(
        email_target='target@email.com',
        chart=chart,
        report_format=ReportDataFormat.CSV,
        name='report_csv_no_query_context',
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_4ih0j8xa() -> Iterator[ReportSchedule]:
    dashboard = db.session.query(Dashboard).first()
    report_schedule = create_report_notification(email_target='target@email.com', dashboard=dashboard)
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_qnc4rbll() -> Iterator[ReportSchedule]:
    dashboard = db.session.query(Dashboard).first()
    report_schedule = create_report_notification(email_target='target@email.com', dashboard=dashboard, force_screenshot=True)
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_r409qxn3() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart)
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_ceoy5m86() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        slack_channel='slack_channel_id', chart=chart, name='report_slack_chartv2'
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_pmx2ayee() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        slack_channel='slack_channel', chart=chart, report_format=ReportDataFormat.CSV
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_8hap9mpp() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    chart.query_context = '{"mock": "query_context"}'
    report_schedule = create_report_notification(
        slack_channel='slack_channel', chart=chart, report_format=ReportDataFormat.TEXT
    )
    yield report_schedule
    cleanup_report_schedule(report_schedule)


@pytest.fixture
def func_67eehlll() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart)
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
def func_4ie9r7b2() -> Iterator[ReportSchedule]:
    chart = db.session.query(Slice).first()
    report_schedule = create_report_notification(
        slack_channel='slack_channel', chart=chart, report_type=ReportScheduleType.ALERT
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


@pytest.fixture(params=['alert1'])
def func_uj5nsbpr(request: pytest.FixtureRequest) -> Iterator[ReportSchedule]:
    param_config = {
        'alert1': {
            'sql': 'SELECT count(*) from test_table',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<", "threshold": 10}',
        }
    }
    chart = db.session.query(Slice).first()
    example_database = get_example_database()
    with func_9aesecuq(example_database):
        report_schedule = create_report_notification(
            slack_channel='slack_channel',
            chart=chart,
            report_type=ReportScheduleType.ALERT,
            database=example_database,
            sql=param_config[request.param]['sql'],
            validator_type=param_config[request.param]['validator_type'],
            validator_config_json=param_config[request.param]['validator_config_json'],
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


@pytest.fixture(params=['alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'alert7', 'alert8'])
def func_zy9dfjyh(request: pytest.FixtureRequest) -> Iterator[ReportSchedule]:
    param_config = {
        'alert1': {
            'sql': 'SELECT 10 as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": ">", "threshold": 9}',
        },
        'alert2': {
            'sql': 'SELECT 10 as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": ">=", "threshold": 10}',
        },
        'alert3': {
            'sql': 'SELECT 10 as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<", "threshold": 11}',
        },
        'alert4': {
            'sql': 'SELECT 10 as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<=", "threshold": 10}',
        },
        'alert5': {
            'sql': 'SELECT 10 as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "!=", "threshold": 11}',
        },
        'alert6': {
            'sql': "SELECT 'something' as metric",
            'validator_type': ReportScheduleValidatorType.NOT_NULL,
            'validator_config_json': '{}',
        },
        'alert7': {
            'sql': 'SELECT {{ 5 + 5 }} as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "!=", "threshold": 11}',
        },
        'alert8': {
            'sql': 'SELECT 55 as metric',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": ">", "threshold": 54.999}',
        },
    }
    chart = db.session.query(Slice).first()
    example_database = get_example_database()
    with func_9aesecuq(example_database):
        report_schedule = create_report_notification(
            email_target='target@email.com',
            chart=chart,
            report_type=ReportScheduleType.ALERT,
            database=example_database,
            sql=param_config[request.param]['sql'],
            validator_type=param_config[request.param]['validator_type'],
            validator_config_json=param_config[request.param]['validator_config_json'],
            force_screenshot=True,
        )
        yield report_schedule
        cleanup_report_schedule(report_schedule)


@pytest.fixture(params=['alert1', 'alert2'])
def func_kgqdzg3t(request: pytest.FixtureRequest) -> Iterator[ReportSchedule]:
    param_config = {
        'alert1': {
            'sql': 'SELECT first, second from test_table',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<", "threshold": 10}',
        },
        'alert2': {
            'sql': 'SELECT first from test_table',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<", "threshold": 10}',
        },
    }
    chart = db.session.query(Slice).first()
    example_database = get_example_database()
    with func_9aesecuq(example_database):
        report_schedule = create_report_notification(
            email_target='target@email.com',
            chart=chart,
            report_type=ReportScheduleType.ALERT,
            database=example_database,
            sql=param_config[request.param]['sql'],
            validator_type=param_config[request.param]['validator_type'],
            validator_config_json=param_config[request.param]['validator_config_json'],
        )
        yield report_schedule
        cleanup_report_schedule(report_schedule)


@pytest.fixture(params=['alert1', 'alert2'])
def func_ka14wyyk(
    request: pytest.FixtureRequest, app_context: AppContext
) -> Iterator[ReportSchedule]:
    param_config = {
        'alert1': {
            'sql': "SELECT 'string' ",
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<", "threshold": 10}',
        },
        'alert2': {
            'sql': 'SELECT first from foo_table',
            'validator_type': ReportScheduleValidatorType.OPERATOR,
            'validator_config_json': '{"op": "<", "threshold": 10}',
        },
    }
    chart = db.session.query(Slice).first()
    example_database = get_example_database()
    with func_9aesecuq(example_database):
        report_schedule = create_report_notification(
            email_target='target@email.com',
            chart=chart,
            report_type=ReportScheduleType.ALERT,
            database=example_database,
            sql=param_config[request.param]['sql'],
            validator_type=param_config[request.param]['validator_type'],
            validator_config_json=param_config[request.param]['validator_config_json'],
            grace_period=60 * 60,
        )
        yield report_schedule
        cleanup_report_schedule(report_schedule)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_cc_and_bcc',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_vcvp9jmu(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart_with_cc_and_bcc: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with screenshot and email cc, bcc options
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(
            TEST_ID,
            create_report_email_chart_with_cc_and_bcc.id,
            datetime.utcnow(),
        ).run()
        notification_targets = func_u0jt1phh(create_report_email_chart_with_cc_and_bcc)
        notification_cctargets = func_uukk02sx(create_report_email_chart_with_cc_and_bcc)
        notification_bcctargets = func_sn4ax80g(create_report_email_chart_with_cc_and_bcc)
        assert (
            f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_with_cc_and_bcc.chart.id}%7D&force=false">Explore in Superset</a>'
            in email_mock.call_args[0][2]
        )
        if notification_targets:
            assert email_mock.call_args[0][0] == notification_targets[0]
        if notification_cctargets:
            expected_cc_targets = [target.strip() for target in notification_cctargets]
            assert email_mock.call_args[1].get('cc', '').split(',') == expected_cc_targets
        if notification_bcctargets:
            expected_bcc_targets = [target.strip() for target in notification_bcctargets]
            assert email_mock.call_args[1].get('bcc', '').split(',') == expected_bcc_targets
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_hlxnx1is(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with screenshot
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart.id, datetime.utcnow()).run()
        notification_targets = func_u0jt1phh(create_report_email_chart)
        assert (
            f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart.chart.id}%7D&force=false">Explore in Superset</a>'
            in email_mock.call_args[0][2]
        )
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_alpha_owner',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch.object(BaseReportState, '_send', side_effect=lambda *args, **kwargs: None)
def func_8zldkgu2(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart_alpha_owner: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with screenshot
    executed as the chart owner
    """
    config_key = 'ALERT_REPORTS_EXECUTORS'
    original_config_value = app.config[config_key]
    app.config[config_key] = [ExecutorType.OWNER]
    username = ''

    def func_hagy7wsm(user: User) -> Any:
        nonlocal username
        username = user.username
        return SCREENSHOT_FILE

    screenshot_mock.side_effect = func_hagy7wsm
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_alpha_owner.id, datetime.utcnow()).run()
        notification_targets = func_u0jt1phh(create_report_email_chart_alpha_owner)
        assert username == 'alpha'
        assert (
            f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_alpha_owner.chart.id}%7D&force=false">Explore in Superset</a>'
            in email_mock.call_args[0][2]
        )
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)
    app.config[config_key] = original_config_value


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_force_screenshot',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_1r0hdf72(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart_force_screenshot: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with screenshot

    In this test ``force_screenshot`` is true, and the screenshot URL should
    reflect that.
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(
            TEST_ID, create_report_email_chart_force_screenshot.id, datetime.utcnow()
        ).run()
        notification_targets = func_u0jt1phh(create_report_email_chart_force_screenshot)
        assert (
            f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_force_screenshot.chart.id}%7D&force=true">Explore in Superset</a>'
            in email_mock.call_args[0][2]
        )
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_d3j5ms0v(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email alert schedule with screenshot
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        notification_targets = func_u0jt1phh(create_alert_email_chart)
        assert (
            f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_alert_email_chart.chart.id}%7D&force=true">Explore in Superset</a>'
            in email_mock.call_args[0][2]
        )
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_xtunr1vk(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule dry run
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    app.config['ALERT_REPORTS_NOTIFICATION_DRY_RUN'] = True
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart.id, datetime.utcnow()).run()
        email_mock.assert_not_called()
    app.config['ALERT_REPORTS_NOTIFICATION_DRY_RUN'] = False


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_csv',
)
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_csv_data')
def func_ul3ow0i4(
    csv_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_csv: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with CSV
    """
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = CSV_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
        notification_targets = func_u0jt1phh(create_report_email_chart_with_csv)
        assert (
            f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_with_csv.chart.id}%7D&force=false">Explore in Superset</a>'
            in email_mock.call_args[0][2]
        )
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['data']
        assert smtp_images[list(smtp_images.keys())[0]] == CSV_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_csv_no_query_context',
)
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_csv_data')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_szrti984(
    screenshot_mock: Mock,
    csv_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_csv_no_query_context: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with CSV (no query context)
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = CSV_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(
            TEST_ID, create_report_email_chart_with_csv_no_query_context.id, datetime.utcnow()
        ).run()
        screenshot_mock.assert_called_once()


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_text',
)
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_dataframe')
def func_qscr9fhj(
    dataframe_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_text: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email report schedule with text
    """
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = json.dumps(
        {
            'result': [
                {
                    'data': {
                        't1': {(0): 'c11', (1): 'c21'},
                        't2': {(0): 'c12', (1): 'c22'},
                        't3__sum': {(0): 'c13', (1): 'c23'},
                    },
                    'colnames': [('t1',), ('t2',), ('t3__sum',)],
                    'indexnames': [(0,), (1,)],
                    'coltypes': [1, 1],
                }
            ]
        }
    ).encode('utf-8')
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_text.id, datetime.utcnow()).run()
        table_html = """<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>t1</th>
      <th>t2</th>
      <th>t3__sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c11</td>
      <td>c12</td>
      <td>c13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c21</td>
      <td>c22</td>
      <td>c23</td>
    </tr>
  </tbody>
</table>"""
        assert table_html in email_mock.call_args[0][2]
        func_3aof9eks(ReportState.SUCCESS)
    dt = datetime(2022, 1, 1).replace(tzinfo=timezone.utc)
    ts = datetime.timestamp(dt) * 1000
    response.read.return_value = json.dumps(
        {
            'result': [
                {
                    'data': {
                        't1': {(0): 'c11', (1): 'c21'},
                        't2__date': {(0): ts, (1): ts},
                        't3__sum': {(0): 'c13', (1): 'c23'},
                    },
                    'colnames': [('t1',), ('t2__date',), ('t3__sum',)],
                    'indexnames': [(0,), (1,)],
                    'coltypes': [1, 2],
                }
            ]
        }
    ).encode('utf-8')
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_text.id, datetime.utcnow()).run()
        table_html = """<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>t1</th>
      <th>t2__date</th>
      <th>t3__sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c11</td>
      <td>2022-01-01</td>
      <td>c13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c21</td>
      <td>2022-01-01</td>
      <td>c23</td>
    </tr>
  </tbody>
</table>"""
        assert table_html in email_mock.call_args[0][2]


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_dashboard',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
def func_ngoo0rc3(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_dashboard: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test dashboard email report schedule
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_dashboard.id, datetime.utcnow()).run()
            notification_targets = func_u0jt1phh(create_report_email_dashboard)
            assert email_mock.call_args[0][0] == notification_targets[0]
            smtp_images = email_mock.call_args[1]['images']
            assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
            func_3aof9eks(ReportState.SUCCESS)
            statsd_mock.assert_called_once_with('reports.email.send.ok', 1)


@pytest.mark.usefixtures(
    'tabbed_dashboard',
)
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERT_REPORT_TABS=True)
def func_055yxou0(
    _email_mock: Mock,
    _screenshot_mock: Mock,
) -> None:
    """
    ExecuteReport Command: Test dashboard email report schedule with tab metadata
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            dashboard = db.session.query(Dashboard).all()[1]
            report_schedule = create_report_notification(
                email_target='target@email.com',
                dashboard=dashboard,
                extra={'dashboard': {'anchor': 'TAB-L2AB'}},
            )
            AsyncExecuteReportScheduleCommand(TEST_ID, report_schedule.id, datetime.utcnow()).run()
            func_3aof9eks(ReportState.SUCCESS)
            statsd_mock.assert_called_once_with('reports.email.send.ok', 1)
            pl = db.session.query(KeyValueEntry).order_by(KeyValueEntry.id.desc()).first()
            value = json.loads(pl.value)
            assert report_schedule.extra['dashboard'] == value['state']
            cleanup_report_schedule(report_schedule)
            reset_key_values()


@pytest.mark.usefixtures(
    'tabbed_dashboard',
)
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERT_REPORT_TABS=False)
def func_6ks96ebb(
    _email_mock: Mock,
    _screenshot_mock: Mock,
) -> None:
    """
    ExecuteReport Command: Test dashboard email report schedule with tab metadata
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            dashboard = db.session.query(Dashboard).all()[1]
            report_schedule = create_report_notification(
                email_target='target@email.com',
                dashboard=dashboard,
                extra={'dashboard': {'anchor': 'TAB-L2AB'}},
            )
            AsyncExecuteReportScheduleCommand(TEST_ID, report_schedule.id, datetime.utcnow()).run()
            func_3aof9eks(ReportState.SUCCESS)
            statsd_mock.assert_called_once_with('reports.email.send.ok', 1)
            permalinks = db.session.query(KeyValueEntry).all()
            assert len(permalinks) == 0
            cleanup_report_schedule(report_schedule)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_dashboard_force_screenshot',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
def func_9xdn0cd6(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_dashboard_force_screenshot: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test dashboard email report schedule
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(
            TEST_ID, create_report_email_dashboard_force_screenshot.id, datetime.utcnow()
        ).run()
        notification_targets = func_u0jt1phh(create_report_email_dashboard_force_screenshot)
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_slack_chart',
)
@patch('superset.commands.report.execute.get_channels_with_search')
@patch('superset.reports.notifications.slack.should_use_v2_api', return_value=True)
@patch('superset.reports.notifications.slackv2.get_slack_client')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_hyqh3ycm(
    screenshot_mock: Mock,
    slack_client_mock: Mock,
    slack_should_use_v2_api_mock: Mock,
    get_channels_with_search_mock: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack report schedule
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    channel_id = 'slack_channel_id'
    get_channels_with_search_mock.return_value = channel_id
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
            assert slack_client_mock.return_value.files_upload_v2.call_args[1]['channel'] == channel_id
            assert slack_client_mock.return_value.files_upload_v2.call_args[1]['file'] == SCREENSHOT_FILE
            assert create_report_slack_chart.recipients[0].recipient_config_json == json.dumps({'target': channel_id})
            assert create_report_slack_chart.recipients[0].type == ReportRecipientType.SLACKV2
            func_3aof9eks(ReportState.SUCCESS)
            assert statsd_mock.call_args_list[0] == call('reports.slack.send.warning', 1)
            assert statsd_mock.call_args_list[1] == call('reports.slack.send.ok', 1)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_slack_chartv2',
)
@patch('superset.commands.report.execute.get_channels_with_search')
@patch('superset.reports.notifications.slack.should_use_v2_api', return_value=True)
@patch('superset.reports.notifications.slackv2.get_slack_client')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_u2i5wd38(
    screenshot_mock: Mock,
    slack_client_mock: Mock,
    slack_should_use_v2_api_mock: Mock,
    get_channels_with_search_mock: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack report schedule
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    channel_id = 'slack_channel_id'
    get_channels_with_search_mock.return_value = channel_id
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
            assert slack_client_mock.return_value.files_upload_v2.call_args[1]['channel'] == channel_id
            assert slack_client_mock.return_value.files_upload_v2.call_args[1]['file'] == SCREENSHOT_FILE
            func_3aof9eks(ReportState.SUCCESS)
            assert statsd_mock.call_args_list[0] == call('reports.slack.send.warning', 1)
            assert statsd_mock.call_args_list[1] == call('reports.slack.send.ok', 1)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_slack_chart',
)
@patch('superset.utils.slack.get_slack_client')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_wetwc8g5(
    screenshot_mock: Mock,
    web_client_mock: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test that all slack errors will
    properly log something
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    slack_errors = [
        BotUserAccessError(),
        SlackRequestError(),
        SlackClientConfigurationError(),
        SlackObjectFormationError(),
        SlackTokenRotationError(api_error='foo'),
        SlackClientNotConnectedError(),
        SlackClientError(),
        SlackApiError(message='foo', response='bar'),
    ]
    for idx, er in enumerate(slack_errors):
        web_client_mock.side_effect = [SlackApiError(None, None), er]
        with pytest.raises(ReportScheduleClientErrorsException):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
        db.session.commit()
    notification_logs_count = func_xbwiqmgs(create_report_slack_chart)
    error_logs = func_mc2fu2ha(create_report_slack_chart)
    assert error_logs.count() == (len(slack_errors) + notification_logs_count) * 2
    assert len([log.error_message for log in error_logs]) == error_logs.count()


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_slack_chart_with_csv',
)
@patch('superset.reports.notifications.slack.should_use_v2_api', return_value=False)
@patch('superset.reports.notifications.slack.get_slack_client')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_csv_data')
def func_uwj0ozvo(
    csv_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    slack_client_mock_class: Mock,
    slack_should_use_v2_api_mock: Mock,
    create_report_slack_chart_with_csv: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack report V1 schedule with CSV
    """
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = CSV_FILE
    notification_targets = func_u0jt1phh(create_report_slack_chart_with_csv)
    channel_name = notification_targets[0]
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_with_csv.id, datetime.utcnow()).run()
        assert slack_client_mock_class.return_value.files_upload.call_args[1]['channels'] == channel_name
        assert slack_client_mock_class.return_value.files_upload.call_args[1]['file'] == CSV_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_slack_chart_with_text',
)
@patch('superset.reports.notifications.slack.should_use_v2_api', return_value=False)
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.slack.get_slack_client')
@patch('superset.utils.csv.get_chart_dataframe')
def func_ow5gg8ds(
    dataframe_mock: Mock,
    slack_client_mock_class: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    slack_should_use_v2_api_mock: Mock,
    create_report_slack_chart_with_text: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack report schedule with text
    """
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = json.dumps(
        {
            'result': [
                {
                    'data': {
                        't1': {(0): 'c11', (1): 'c21'},
                        't2': {(0): 'c12', (1): 'c22'},
                        't3__sum': {(0): 'c13', (1): 'c23'},
                    },
                    'colnames': [('t1',), ('t2',), ('t3__sum',)],
                    'indexnames': [(0,), (1,)],
                    'coltypes': [1, 1, 0],
                }
            ]
        }
    ).encode('utf-8')
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_with_text.id, datetime.utcnow()).run()
        table_markdown = """|    | t1   | t2   | t3__sum   |
|---:|:-----|:-----|:----------|
|  0 | c11  | c12  | c13       |
|  1 | c21  | c22  | c23       |"""
        assert table_markdown in slack_client_mock_class.return_value.chat_postMessage.call_args[1]['text']
        assert (
            f'<http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_slack_chart_with_text.chart.id}%7D&force=false|Explore in Superset>'
            in slack_client_mock_class.return_value.chat_postMessage.call_args[1]['text']
        )
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'create_report_slack_chart',
)
def func_rp3902jd(create_report_slack_chart: ReportSchedule) -> None:
    """
    ExecuteReport Command: Test report schedule not found
    """
    max_id = db.session.query(func.max(ReportSchedule.id)).scalar()
    with pytest.raises(ReportScheduleNotFoundError):
        AsyncExecuteReportScheduleCommand(TEST_ID, max_id + 1, datetime.utcnow()).run()


@pytest.mark.usefixtures(
    'create_report_slack_chart_working',
)
def func_5milm7cn(create_report_slack_chart_working: ReportSchedule) -> None:
    """
    ExecuteReport Command: Test report schedule still working
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises(ReportSchedulePreviousWorkingError):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_working.id, datetime.utcnow()).run()
        func_3aof9eks(ReportState.WORKING, error_message=ReportSchedulePreviousWorkingError.message)
        assert create_report_slack_chart_working.last_state == ReportState.WORKING


@pytest.mark.usefixtures(
    'create_report_slack_chart_working',
)
def func_o3a29q4n(create_report_slack_chart_working: ReportSchedule) -> None:
    """
    ExecuteReport Command: Test report schedule still working but should timed out
    """
    current_time = create_report_slack_chart_working.last_eval_dttm + timedelta(
        seconds=create_report_slack_chart_working.working_timeout + 1
    )
    with freeze_time(current_time):
        with pytest.raises(ReportScheduleWorkingTimeoutError):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_working.id, datetime.utcnow()).run()
    logs = db.session.query(ReportExecutionLog).all()
    assert len(logs) == 2
    assert ReportScheduleWorkingTimeoutError.message in [log.error_message for log in logs]
    assert create_report_slack_chart_working.last_state == ReportState.ERROR


@pytest.mark.usefixtures(
    'create_alert_slack_chart_success',
)
def func_jx7oohjh(create_alert_slack_chart_success: ReportSchedule) -> None:
    """
    ExecuteReport Command: Test report schedule on success to grace
    """
    current_time = create_alert_slack_chart_success.last_eval_dttm + timedelta(
        seconds=create_alert_slack_chart_success.grace_period - 10
    )
    with freeze_time(current_time):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_slack_chart_success.id, datetime.utcnow()).run()
    db.session.commit()
    assert create_alert_slack_chart_success.last_state == ReportState.GRACE


@pytest.mark.usefixtures(
    'create_alert_slack_chart_grace',
)
@patch('superset.utils.slack.WebClient.files_upload')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch('superset.reports.notifications.slack.get_slack_client')
def func_9ycn8zsb(
    slack_client_mock_class: Mock,
    screenshot_mock: Mock,
    file_upload_mock: Mock,
    create_alert_slack_chart_grace: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test report schedule on grace to noop
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    current_time = create_alert_slack_chart_grace.last_eval_dttm + timedelta(
        seconds=create_alert_slack_chart_grace.grace_period + 1
    )
    notification_targets = func_u0jt1phh(create_alert_slack_chart_grace)
    channel_id = 'channel_id'
    slack_client_mock_class.return_value.conversations_list.return_value = {'channels': [{'id': channel_id, 'name': notification_targets[0]}]}
    with freeze_time(current_time):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_slack_chart_grace.id, datetime.utcnow()).run()
    db.session.commit()
    assert create_alert_slack_chart_grace.last_state == ReportState.SUCCESS


@pytest.mark.usefixtures(
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
def func_xw5yvqpf(
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test that all alerts apply a SQL limit to stmts
    """
    screenshot_mock = Mock()
    screenshot_mock.return_value = SCREENSHOT_FILE
    with patch.object(
        create_alert_email_chart.database.db_engine_spec,
        'execute',
        return_value=None,
    ) as execute_mock:
        with patch.object(
            create_alert_email_chart.database.db_engine_spec,
            'fetch_data',
            return_value=[],
        ):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
            assert 'LIMIT 2' in execute_mock.call_args[0][1]


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_dashboard',
)
@patch('superset.daos.report.ReportScheduleDAO.bulk_delete_logs')
def func_lalh733r(
    bulk_delete_logs: Mock,
    create_report_email_dashboard: ReportSchedule,
) -> None:
    from celery.exceptions import SoftTimeLimitExceeded

    bulk_delete_logs.side_effect = SoftTimeLimitExceeded()
    with pytest.raises(SoftTimeLimitExceeded) as excinfo:
        AsyncPruneReportScheduleLogCommand().run()
    assert str(excinfo.value) == 'SoftTimeLimitExceeded()'


@patch('superset.commands.report.execute.logger')
@patch('superset.commands.report.execute.create_notification')
def func_aububutk(
    notification_mock: Mock,
    logger_mock: Mock,
) -> None:
    notification_content = 'I am some content'
    recipients = ['test@foo.com']
    notification_mock.return_value.send.side_effect = NotificationParamException()
    with pytest.raises(ReportScheduleClientErrorsException):
        BaseReportState._send(BaseReportState, notification_content, recipients)
    assert issubclass(type(logger_mock.warning.call_args), Mock)
    logger_mock.warning.assert_called_with(
        "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.WARNING: 'warning'>, extra=None)"
    )


@patch('superset.commands.report.execute.logger')
@patch('superset.commands.report.execute.create_notification')
def func_nsyq5iy3(
    notification_mock: Mock,
    logger_mock: Mock,
) -> None:
    notification_content = 'I am some content'
    recipients = ['test@foo.com', 'test2@bar.com']
    notification_mock.return_value.send.side_effect = [NotificationParamException(), NotificationError()]
    with pytest.raises(ReportScheduleSystemErrorsException):
        BaseReportState._send(BaseReportState, notification_content, recipients)
    logger_mock.warning.assert_has_calls([
        call(
            "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.WARNING: 'warning'>, extra=None)"
        ),
        call(
            "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra=None)"
        ),
    ])


@patch('superset.commands.report.execute.logger')
@patch('superset.commands.report.execute.create_notification')
def func_qja2uj8q(
    notification_mock: Mock,
    logger_mock: Mock,
) -> None:
    notification_content = 'I am some content'
    recipients = ['test@foo.com']
    notification_mock.return_value.send.side_effect = NotificationError()
    with pytest.raises(ReportScheduleSystemErrorsException):
        BaseReportState._send(BaseReportState, notification_content, recipients)
    logger_mock.warning.assert_called_with(
        "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra=None)"
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_dashboard',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=True)
def func_n9426i80(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack alert
    """
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        notification_targets = func_u0jt1phh(create_alert_email_chart)
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=False)
def func_ol9b90nz(
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack alert
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        notification_targets = func_u0jt1phh(create_alert_email_chart)
        assert email_mock.call_args[0][0] == notification_targets[0]
        assert email_mock.call_args[1]['images'] == {}
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_slack_chart',
)
@patch('superset.utils.slack.WebClient')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch('superset.reports.notifications.slack.get_slack_client')
def func_5crokhls(
    screenshot_mock: Mock,
    slack_client_mock_class: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart slack alert (slack token callable)
    """
    notification_targets = func_u0jt1phh(create_report_slack_chart)
    channel_name = notification_targets[0]
    channel_id = 'channel_id'
    slack_client_mock_class.return_value.conversations_list.return_value = {'channels': [{'id': channel_id, 'name': channel_name}]}
    app.config['SLACK_API_TOKEN'] = Mock(return_value='cool_code')
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
        app.config['SLACK_API_TOKEN'].assert_called()
        slack_client_mock_class.assert_called_with(token='cool_code', proxy='')
        func_3aof9eks(ReportState.SUCCESS)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_no_alert_email_chart',
)
def func_opymkrwa(
    create_no_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email no alert
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_no_alert_email_chart.id, datetime.utcnow()).run()
    func_3aof9eks(ReportState.NOOP)


@pytest.mark.usefixtures(
    'app_context',
)
def func_8pgni16w(
    create_mul_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test chart email multiple rows
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryMultipleRowsError, AlertQueryMultipleColumnsError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_mul_alert_email_chart.id, datetime.utcnow()).run()


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
def func_7qzkwmdp(
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test soft timeout on alert queries
    """
    from celery.exceptions import SoftTimeLimitExceeded
    from superset.commands.report.exceptions import AlertQueryTimeout

    with patch.object(
        create_alert_email_chart.database.db_engine_spec,
        'execute',
        return_value=None,
    ) as execute_mock:
        execute_mock.side_effect = SoftTimeLimitExceeded()
        with pytest.raises(AlertQueryTimeout):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
    func_u0jt1phh(create_alert_email_chart)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    func_3aof9eks(
        ReportState.ERROR,
        error_message='A timeout occurred while executing the query.',
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=True)
def func_llctg392(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test soft timeout on screenshot
    """
    from celery.exceptions import SoftTimeLimitExceeded
    from superset.commands.report.exceptions import AlertQueryTimeout

    screenshot_mock.side_effect = SoftTimeLimitExceeded()
    with pytest.raises(ReportScheduleScreenshotTimeout):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    func_3aof9eks(
        ReportState.ERROR,
        error_message='A timeout occurred while taking a screenshot.',
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_csv',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_csv_data')
def func_nni6s7jf(
    csv_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_csv: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test fail on generating csv
    """
    from celery.exceptions import SoftTimeLimitExceeded
    from superset.commands.report.exceptions import ReportScheduleCsvTimeout

    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.side_effect = SoftTimeLimitExceeded()
    with pytest.raises(ReportScheduleCsvTimeout):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
    func_u0jt1phh(create_report_email_chart_with_csv)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    func_3aof9eks(
        ReportState.ERROR,
        error_message='A timeout occurred while generating a csv.',
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_csv',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_csv_data')
def func_bg7endyw(
    csv_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_csv: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test fail on generating csv
    """
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = None
    with pytest.raises(ReportScheduleCsvFailedError):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
    func_u0jt1phh(create_report_email_chart_with_csv)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    func_3aof9eks(
        ReportState.ERROR,
        error_message='Report Schedule execution failed when generating a csv.',
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def func_exjk8qhl(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test soft timeout on screenshot
    """
    from superset.commands.report.exceptions import ReportScheduleScreenshotFailedError

    screenshot_mock.side_effect = Exception('Unexpected error')
    with pytest.raises(ReportScheduleScreenshotFailedError):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart.id, datetime.utcnow()).run()
    func_u0jt1phh(create_report_email_chart)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    func_3aof9eks(
        ReportState.ERROR,
        error_message='Failed taking a screenshot Unexpected error',
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_chart_with_csv',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_csv_data')
def func_6coworwk(
    csv_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    email_mock: Mock,
    create_report_email_chart_with_csv: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test error on csv
    """
    from superset.commands.report.exceptions import ReportScheduleCsvFailedError

    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 500
    with pytest.raises(ReportScheduleCsvFailedError):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
    func_u0jt1phh(create_report_email_chart_with_csv)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    func_3aof9eks(
        ReportState.ERROR,
        error_message='Failed generating csv <urlopen error 500>',
    )


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=False)
def func_hfmb21x6(
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test soft timeout on screenshot
    """
    AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
    notification_targets = func_u0jt1phh(create_alert_email_chart)
    assert email_mock.call_args[0][0] == notification_targets[0]
    assert email_mock.call_args[1]['images'] == {}
    func_3aof9eks(ReportState.SUCCESS)


@patch('superset.reports.notifications.email.send_email_smtp')
def func_f7uaqofg(
    email_mock: Mock,
    create_invalid_sql_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test alert with invalid SQL statements
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
        func_3aof9eks(ReportState.ERROR)


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_alert_email_chart',
)
@patch('superset.reports.notifications.email.send_email_smtp')
def func_3abyqvvr(
    email_mock: Mock,
    create_invalid_sql_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test alert grace period on error
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
        assert func_xbwiqmgs(create_invalid_sql_alert_email_chart) == 1
    with freeze_time('2020-01-01T00:30:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert func_xbwiqmgs(create_invalid_sql_alert_email_chart) == 1
    with freeze_time('2020-01-01T01:30:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert func_xbwiqmgs(create_invalid_sql_alert_email_chart) == 2
    create_invalid_sql_alert_email_chart.sql = 'SELECT 1 AS metric'
    create_invalid_sql_alert_email_chart.grace_period = 0
    db.session.commit()
    with freeze_time('2020-01-01T00:31:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
    create_invalid_sql_alert_email_chart.sql = "SELECT 'first'"
    create_invalid_sql_alert_email_chart.grace_period = 10
    db.session.commit()
    with freeze_time('2020-01-01T00:32:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert func_xbwiqmgs(create_invalid_sql_alert_email_chart) == 2


@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=True)
def func_ctdxb9fi(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None:
    """
    ExecuteReport Command: Test alert grace period on error
    """
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert func_xbwiqmgs(create_alert_email_chart) == 1
    with freeze_time('2020-01-01T00:30:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert func_xbwiqmgs(create_alert_email_chart) == 1
    create_alert_email_chart.sql = 'SELECT 1 AS metric'
    create_alert_email_chart.grace_period = 0
    db.session.commit()
    with freeze_time('2020-01-01T00:31:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
    create_alert_email_chart.sql = "SELECT 'first'"
    create_alert_email_chart.grace_period = 10
    db.session.commit()
    with freeze_time('2020-01-01T00:32:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert func_xbwiqmgs(create_alert_email_chart) == 2


@pytest.mark.usefixtures(
    'load_birth_names_dashboard_with_slices',
    'create_report_email_dashboard',
)
@patch('superset.commands.report.execute.logger')
@patch('superset.commands.report.execute.create_notification')
def func_aububutk(
    notification_mock: Mock,
    logger_mock: Mock,
) -> None:
    notification_content = 'I am some content'
    recipients = ['test@foo.com']
    notification_mock.return_value.send.side_effect = NotificationParamException()
    with pytest.raises(ReportScheduleClientErrorsException):
        BaseReportState._send(BaseReportState, notification_content, recipients)
    logger_mock.warning.assert_called_with(
        "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.WARNING: 'warning'>, extra=None)"
    )


@patch('superset.commands.report.execute.logger')
@patch('superset.commands.report.execute.create_notification')
def func_nsyq5iy3(
    notification_mock: Mock,
    logger_mock: Mock,
) -> None:
    notification_content = 'I am some content'
    recipients = ['test@foo.com', 'test2@bar.com']
    notification_mock.return_value.send.side_effect = [NotificationParamException(), NotificationError()]
    with pytest.raises(ReportScheduleSystemErrorsException):
        BaseReportState._send(BaseReportState, notification_content, recipients)
    logger_mock.warning.assert_has_calls([
        call(
            "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.WARNING: 'warning'>, extra=None)"
        ),
        call(
            "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra=None)"
        ),
    ])


@patch('superset.commands.report.execute.logger')
@patch('superset.commands.report.execute.create_notification')
def func_qja2uj8q(
    notification_mock: Mock,
    logger_mock: Mock,
) -> None:
    notification_content = 'I am some content'
    recipients = ['test@foo.com']
    notification_mock.return_value.send.side_effect = NotificationError()
    with pytest.raises(ReportScheduleSystemErrorsException):
        BaseReportState._send(BaseReportState, notification_content, recipients)
    logger_mock.warning.assert_called_with(
        "SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra=None)"
    )
