    def pre_receive(self, alert: 'Alert', **kwargs: Any) -> 'Alert':
    def post_receive(self, alert: 'Alert', **kwargs: Any) -> 'Alert':
    def status_change(self, alert: 'Alert', status: str, text: str, **kwargs: Any) -> None:
    def take_action(self, alert: 'Alert', action: str, text: str, **kwargs: Any) -> 'Alert':
    def delete(self, alert: 'Alert', **kwargs: Any) -> bool:
