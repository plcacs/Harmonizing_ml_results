from typing import Any, AnyStr, Dict

class OutgoingWebhookServiceInterface(abc.ABC):

    def __init__(self, token: str, user_profile: Any, service_name: str) -> None:
        self.token = token
        self.user_profile = user_profile
        self.service_name = service_name
        self.session = OutgoingSession(role='webhook', timeout=settings.OUTGOING_WEBHOOK_TIMEOUT_SECONDS, headers={'User-Agent': 'ZulipOutgoingWebhook/' + ZULIP_VERSION})

    @abc.abstractmethod
    def make_request(self, base_url: str, event: Dict[str, Any], realm: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def process_success(self, response_json: Dict[str, Any]) -> Any:
        raise NotImplementedError

class GenericOutgoingWebhookService(OutgoingWebhookServiceInterface):

    @override
    def make_request(self, base_url: str, event: Dict[str, Any], realm: Any) -> Any:
        ...

    @override
    def process_success(self, response_json: Dict[str, Any]) -> Any:
        ...

class SlackOutgoingWebhookService(OutgoingWebhookServiceInterface):

    @override
    def make_request(self, base_url: str, event: Dict[str, Any], realm: Any) -> Any:
        ...

    @override
    def process_success(self, response_json: Dict[str, Any]) -> Any:
        ...

def get_service_interface_class(interface: str) -> Any:
    ...

def get_outgoing_webhook_service_handler(service: Any) -> Any:
    ...

def send_response_message(bot_id: int, message_info: Dict[str, Any], response_data: Dict[str, Any]) -> None:
    ...

def fail_with_message(event: Dict[str, Any], failure_message: str) -> None:
    ...

def get_message_url(event: Dict[str, Any]) -> str:
    ...

def notify_bot_owner(event: Dict[str, Any], status_code: int = None, response_content: str = None, failure_message: str = None, exception: Exception = None) -> None:
    ...

def request_retry(event: Dict[str, Any], failure_message: str = None) -> None:
    ...

def process_success_response(event: Dict[str, Any], service_handler: Any, response: Any) -> None:
    ...

def do_rest_call(base_url: str, event: Dict[str, Any], service_handler: Any) -> Any:
    ...
