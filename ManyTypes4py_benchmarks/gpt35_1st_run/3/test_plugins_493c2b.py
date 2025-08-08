    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
    def post_receive(self, alert: Alert) -> Alert:
    def status_change(self, alert: Alert, status: str, text: str) -> Tuple[Alert, str, str]:
    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
    def post_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Alert:
    def take_note(self, alert: Alert, text: str, **kwargs: Any) -> Tuple[Alert, str]:
    def delete(self, alert: Alert, **kwargs: Any) -> bool:
