def assign_customer(wanted: Optional[str] = None, permission: Scope = Scope.admin_alerts) -> Optional[str]:
def process_alert(alert: Alert) -> Alert:
def process_action(alert: Alert, action: str, text: str, timeout: Optional[int] = None, post_action: bool = False) -> Tuple[Alert, str, str, Optional[int]]:
def process_note(alert: Alert, text: str) -> Tuple[Alert, str]:
def process_status(alert: Alert, status: str, text: str) -> Tuple[Alert, str, str]:
def process_delete(alert: Alert) -> bool:
