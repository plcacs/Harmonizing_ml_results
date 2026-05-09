from typing import Any, Dict, List

class MockSMTP(MailNotificationService):
    """Test SMTP object that doesn't need a working server."""

    def _send_email(self, msg: Any, recipients: List[str]) -> Tuple[str, List[str]]:
        """Just return msg string and recipients for testing."""
        return (msg.as_string(), recipients)

async def test_reload_notify(hass: HomeAssistant) -> None:
    """Verify we can reload the notify service."""
    # ...

@pytest.fixture
def message() -> MockSMTP:
    """Return MockSMTP object with test data."""
    return MockSMTP('localhost', 25, 5, 'test@test.com', 1, 'testuser', 'testpass', ['recip1@example.com', 'testrecip@test.com'], 'Home Assistant', 0, True)

HTML = '\n        <!DOCTYPE html>\n        <html lang="en" xmlns="http://www.w3.org/1999/xhtml">\n            <head><meta charset="UTF-8"></head>\n            <body>\n              <div>\n                <h1>Intruder alert at apartment!!</h1>\n              </div>\n              <div>\n                <img alt="tests/testing_config/notify/test.jpg" src="cid:tests/testing_config/notify/test.jpg"/>\n              </div>\n            </body>\n        </html>'
EMAIL_DATA: List[Tuple[str, Dict[str, List[str]], str]] = [('Test msg', {'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/mixed'), ...]

@pytest.mark.parametrize(('message_data', 'data', 'content_type'), EMAIL_DATA, ids=['Tests when sending text message and images.', ...])
def test_send_message(hass: HomeAssistant, message_data: str, data: Dict[str, List[str]], content_type: str, message: MockSMTP) -> None:
    """Verify if we can send messages of all types correctly."""
    # ...

def test_send_text_message(hass: HomeAssistant, message: MockSMTP) -> None:
    """Verify if we can send simple text message."""
    # ...

@pytest.mark.parametrize('target', [None, 'target@example.com'], ids=['Verify we can send email to default recipient.', 'Verify email recipient can be overwritten by target arg.'])
def test_send_target_message(target: str, hass: HomeAssistant, message: MockSMTP) -> None:
    """Verify if we can send email to correct recipient."""
    # ...
