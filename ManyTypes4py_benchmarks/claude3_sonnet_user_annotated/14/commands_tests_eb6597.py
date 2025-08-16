from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
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
    load_birth_names_dashboard_with_slices,  # noqa: F401
    load_birth_names_data,  # noqa: F401
)
from tests.integration_tests.fixtures.tabbed_dashboard import (
    tabbed_dashboard,  # noqa: F401
)
from tests.integration_tests.fixtures.world_bank_dashboard import (
    load_world_bank_dashboard_with_slices_module_scope,  # noqa: F401
    load_world_bank_data,  # noqa: F401
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


def assert_log(state: str, error_message: Optional[str] = None) -> None:
    db.session.commit()
    logs = db.session.query(ReportExecutionLog).all()

    if state == ReportState.ERROR:
        # On error we send an email
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
def create_test_table_context(database: Database) -> Any:
    with database.get_sqla_engine() as engine:
        engine.execute(
            "CREATE TABLE IF NOT EXISTS test_table AS SELECT 1 as first, 2 as second"
        )
        engine.execute("INSERT INTO test_table (first, second) VALUES (1, 2)")
        engine.execute("INSERT INTO test_table (first, second) VALUES (3, 4)")

    yield db.session
    with database.get_sqla_engine() as engine:
        engine.execute("DROP TABLE test_table")
