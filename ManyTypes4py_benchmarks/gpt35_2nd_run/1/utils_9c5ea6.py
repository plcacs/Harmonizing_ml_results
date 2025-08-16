def insert_report_schedule(type: str, name: str, crontab: str, owners: list[User], timezone: Optional[str] = None, sql: Optional[str] = None, description: Optional[str] = None, chart: Optional[Slice] = None, dashboard: Optional[Dashboard] = None, database: Optional[Database] = None, validator_type: Optional[str] = None, validator_config_json: Optional[str] = None, log_retention: Optional[int] = None, last_state: Optional[ReportState] = None, grace_period: Optional[int] = None, recipients: Optional[list[ReportRecipients]] = None, report_format: Optional[ReportDataFormat] = None, logs: Optional[list[ReportExecutionLog]] = None, extra: Any = None, force_screenshot: bool = False) -> ReportSchedule:

def create_report_notification(email_target: Optional[str] = None, slack_channel: Optional[str] = None, chart: Optional[Slice] = None, dashboard: Optional[Dashboard] = None, database: Optional[Database] = None, sql: Optional[str] = None, report_type: ReportScheduleType = ReportScheduleType.REPORT, validator_type: Optional[str] = None, validator_config_json: Optional[str] = None, grace_period: Optional[int] = None, report_format: Optional[ReportDataFormat] = None, name: Optional[str] = None, extra: Any = None, force_screenshot: bool = False, owners: Optional[list[User]] = None, ccTarget: Optional[str] = None, bccTarget: Optional[str] = None) -> ReportSchedule:

def cleanup_report_schedule(report_schedule: Optional[ReportSchedule] = None) -> None:

@contextmanager
def create_dashboard_report(dashboard: Dashboard, extra: Any, **kwargs) -> Generator[ReportSchedule, None, None]:
