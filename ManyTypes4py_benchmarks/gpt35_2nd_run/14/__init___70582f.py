    def pre_receive(self, alert: 'Alert', **kwargs: Any) -> None:
    def post_receive(self, alert: 'Alert', **kwargs: Any) -> None:
    def status_change(self, alert: 'Alert', status: str, text: str, **kwargs: Any) -> None:
    def take_action(self, alert: 'Alert', action: str, text: str, **kwargs: Any) -> None:
    def post_action(self, alert: 'Alert', action: str, text: str, **kwargs: Any) -> None:
    def take_note(self, alert: 'Alert', text: str, **kwargs: Any) -> None:
    def delete(self, alert: 'Alert', **kwargs: Any) -> None:

    def get_config(key: str, default: Optional[Any] = None, type: Optional[type] = None, **kwargs: Any) -> Any:
