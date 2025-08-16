    def pre_receive(self, alert: 'Alert', **kwargs: Any) -> 'Alert':
    def post_receive(self, alert: 'Alert', **kwargs: Any) -> Optional['Alert']:
    def status_change(self, alert: 'Alert', status: str, text: str, **kwargs: Any) -> None:
    def take_action(self, alert: 'Alert', action: ChangeType, text: str, **kwargs: Any) -> tuple['Alert', ChangeType, str, int]:
    def take_note(self, alert: 'Alert', text: str, **kwargs: Any) -> None:
    def delete(self, alert: 'Alert', **kwargs: Any) -> None:
