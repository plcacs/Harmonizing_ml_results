from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch
import pytest
from homeassistant import config as hass_config
from homeassistant.components import notify
from homeassistant.components.smtp.const import DOMAIN
from homeassistant.components.smtp.notify import MailNotificationService
from homeassistant.const import SERVICE_RELOAD
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.setup import async_setup_component
from tests.common import get_fixture_path

class MockSMTP(MailNotificationService):
    def _send_email(self, msg, recipients) -> Tuple[str, List[str]]:
        return (msg.as_string(), recipients)

async def test_reload_notify(hass: HomeAssistant):
    ...

@pytest.fixture
def message() -> MockSMTP:
    ...

HTML: str = '\n        <!DOCTYPE html>\n        <html lang="en" xmlns="http://www.w3.org/1999/xhtml">\n            <head><meta charset="UTF-8"></head>\n            <body>\n              <div>\n                <h1>Intruder alert at apartment!!</h1>\n              </div>\n              <div>\n                <img alt="tests/testing_config/notify/test.jpg" src="cid:tests/testing_config/notify/test.jpg"/>\n              </div>\n            </body>\n        </html>'
EMAIL_DATA: List[Tuple[str, dict, str]] = [('Test msg', {'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/mixed'), ('Test msg', {'html': HTML, 'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/related'), ('Test msg', {'html': HTML, 'images': ['tests/testing_config/notify/test_not_exists.jpg']}, 'Content-Type: multipart/related'), ('Test msg', {'html': HTML, 'images': ['tests/testing_config/notify/test.pdf']}, 'Content-Type: multipart/related')]

@pytest.mark.parametrize(('message_data', 'data', 'content_type'), EMAIL_DATA, ids=['Tests when sending text message and images.', 'Tests when sending text message, HTML Template and images.', 'Tests when image does not exist at mentioned location.', 'Tests when image type cannot be detected or is of wrong type.'])
def test_send_message(hass: HomeAssistant, message_data: str, data: dict, content_type: str, message: MockSMTP):
    ...

@pytest.mark.parametrize(('message_data', 'data', 'content_type'), [('Test msg', {'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/mixed')])
def test_sending_insecure_files_fails(hass: HomeAssistant, message_data: str, data: dict, content_type: str, message: MockSMTP):
    ...

def test_send_text_message(hass: HomeAssistant, message: MockSMTP):
    ...

@pytest.mark.parametrize('target', [None, 'target@example.com'], ids=['Verify we can send email to default recipient.', 'Verify email recipient can be overwritten by target arg.'])
def test_send_target_message(target: str, hass: HomeAssistant, message: MockSMTP):
    ...
