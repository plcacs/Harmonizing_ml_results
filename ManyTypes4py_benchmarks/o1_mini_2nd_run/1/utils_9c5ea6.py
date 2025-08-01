from contextlib import contextmanager
from typing import Any, Optional, List, Dict, Generator
from uuid import uuid4
from flask_appbuilder.security.sqla.models import User
from superset import db, security_manager
from superset.key_value.models import KeyValueEntry
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.reports.models import (
    ReportDataFormat,
    ReportExecutionLog,
    ReportRecipients,
    ReportRecipientType,
    ReportSchedule,
    ReportScheduleType,
    ReportState
)
from superset.utils import json
from superset.utils.core import override_user
from tests.integration_tests.test_app import app
from tests.integration_tests.utils import read_fixture

TEST_ID: str = str(uuid4())
CSV_FILE: str = read_fixture('trends.csv')
SCREENSHOT_FILE: str = read_fixture('sample.png')
DEFAULT_OWNER_EMAIL: str = 'admin@fab.org'

def insert_report_schedule(
    type: ReportScheduleType,
    name: str,
    crontab: str,
    owners: Optional[List[User]],
    timezone: Optional[str] = None,
    sql: Optional[str] = None,
    description: Optional[str] = None,
    chart: Optional[Slice] = None,
    dashboard: Optional[Dashboard] = None,
    database: Optional[Database] = None,
    validator_type: Optional[str] = None,
    validator_config_json: Optional[str] = None,
    log_retention: Optional[int] = None,
    last_state: Optional[ReportState] = None,
    grace_period: Optional[int] = None,
    recipients: Optional[List[ReportRecipients]] = None,
    report_format: Optional[ReportDataFormat] = None,
    logs: Optional[List[ReportExecutionLog]] = None,
    extra: Optional[Dict[str, Any]] = None,
    force_screenshot: bool = False
) -> ReportSchedule:
    owners = owners or []
    recipients = recipients or []
    logs = logs or []
    last_state = last_state or ReportState.NOOP
    with override_user(owners[0]):
        report_schedule = ReportSchedule(
            type=type,
            name=name,
            crontab=crontab,
            timezone=timezone,
            sql=sql,
            description=description,
            chart=chart,
            dashboard=dashboard,
            database=database,
            owners=owners,
            validator_type=validator_type,
            validator_config_json=validator_config_json,
            log_retention=log_retention,
            grace_period=grace_period,
            recipients=recipients,
            logs=logs,
            last_state=last_state,
            report_format=report_format,
            extra=extra,
            force_screenshot=force_screenshot
        )
    db.session.add(report_schedule)
    db.session.commit()
    return report_schedule

def create_report_notification(
    email_target: Optional[str] = None,
    slack_channel: Optional[str] = None,
    chart: Optional[Slice] = None,
    dashboard: Optional[Dashboard] = None,
    database: Optional[Database] = None,
    sql: Optional[str] = None,
    report_type: ReportScheduleType = ReportScheduleType.REPORT,
    validator_type: Optional[str] = None,
    validator_config_json: Optional[str] = None,
    grace_period: Optional[int] = None,
    report_format: Optional[ReportDataFormat] = None,
    name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    force_screenshot: bool = False,
    owners: Optional[List[User]] = None,
    ccTarget: Optional[str] = None,
    bccTarget: Optional[str] = None
) -> ReportSchedule:
    if not owners:
        owners = [db.session.query(security_manager.user_model).filter_by(email=DEFAULT_OWNER_EMAIL).one_or_none()]
    if slack_channel:
        recipient = ReportRecipients(
            type=ReportRecipientType.SLACK,
            recipient_config_json=json.dumps({'target': slack_channel})
        )
    else:
        recipient = ReportRecipients(
            type=ReportRecipientType.EMAIL,
            recipient_config_json=json.dumps({'target': email_target, 'ccTarget': ccTarget, 'bccTarget': bccTarget})
        )
    if name is None:
        name = 'report_with_csv' if report_format else 'report'
    report_schedule = insert_report_schedule(
        type=report_type,
        name=name,
        crontab='0 9 * * *',
        owners=owners,
        timezone=None,
        sql=sql,
        description='Daily report',
        chart=chart,
        dashboard=dashboard,
        database=database,
        validator_type=validator_type,
        validator_config_json=validator_config_json,
        log_retention=None,
        last_state=None,
        grace_period=grace_period,
        recipients=[recipient],
        report_format=report_format or ReportDataFormat.PNG,
        logs=None,
        extra=extra,
        force_screenshot=force_screenshot
    )
    return report_schedule

def cleanup_report_schedule(
    report_schedule: Optional[ReportSchedule] = None
) -> None:
    if report_schedule:
        db.session.query(ReportExecutionLog).filter(ReportExecutionLog.report_schedule == report_schedule).delete()
        db.session.query(ReportRecipients).filter(ReportRecipients.report_schedule == report_schedule).delete()
        db.session.delete(report_schedule)
    else:
        db.session.query(ReportExecutionLog).delete()
        db.session.query(ReportRecipients).delete()
        db.session.query(ReportSchedule).delete()
    db.session.commit()

@contextmanager
def create_dashboard_report(
    dashboard: Dashboard,
    extra: Any,
    **kwargs: Any
) -> Generator[ReportSchedule, None, None]:
    report_schedule = create_report_notification(
        email_target='target@example.com',
        dashboard=dashboard,
        extra={'dashboard': extra},
        **kwargs
    )
    error: Optional[Exception] = None
    try:
        yield report_schedule
    except Exception as ex:
        error = ex
    cleanup_report_schedule(report_schedule)
    if error:
        raise error

def reset_key_values() -> None:
    db.session.query(KeyValueEntry).delete()
    db.session.commit()
