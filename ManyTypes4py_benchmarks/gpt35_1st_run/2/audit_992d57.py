    def init_app(self, app: Flask) -> None:
    def _log_response(self, app: Flask, category: str, event: str, message: str, user: str, customers: List[str], scopes: List[str], resource_id: str, type: str, request: Any, **extra: Any) -> None:
    def _webhook_response(self, app: Flask, category: str, event: str, message: str, user: str, customers: List[str], scopes: List[str], resource_id: str, type: str, request: Any, **extra: Any) -> None:
    def admin_log_response(self, app: Flask, **kwargs: Any) -> None:
    def admin_webhook_response(self, app: Flask, **kwargs: Any) -> None:
    def write_log_response(self, app: Flask, **kwargs: Any) -> None:
    def write_webhook_response(self, app: Flask, **kwargs: Any) -> None:
    def auth_log_response(self, app: Flask, **kwargs: Any) -> None:
    def auth_webhook_response(self, app: Flask, **kwargs: Any) -> None:
    @staticmethod
    def _fmt(app: Flask, category: str, event: str, message: str, user: str, customers: List[str], scopes: List[str], resource_id: str, type: str, request: Any, **extra: Any) -> str:
