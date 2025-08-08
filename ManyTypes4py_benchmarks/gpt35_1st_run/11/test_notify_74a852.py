from homeassistant.core import HomeAssistant
from homeassistant.components.smtp.notify import MailNotificationService
from typing import List, Tuple, Dict

async def test_reload_notify(hass: HomeAssistant) -> None:
    ...

def test_send_message(hass: HomeAssistant, message_data: str, data: Dict[str, List[str]], content_type: str, message: MailNotificationService) -> None:
    ...

def test_sending_insecure_files_fails(hass: HomeAssistant, message_data: str, data: Dict[str, List[str]], content_type: str, message: MailNotificationService) -> None:
    ...

def test_send_text_message(hass: HomeAssistant, message: MailNotificationService) -> None:
    ...

def test_send_target_message(target: str, hass: HomeAssistant, message: MailNotificationService) -> None:
    ...
