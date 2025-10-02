"""The tests for the notify smtp platform."""
from pathlib import Path
import re
from unittest.mock import patch
from typing import Any, Tuple, List, Optional
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
from email.message import Message


class MockSMTP(MailNotificationService):
    """Test SMTP object that doesn't need a working server."""

    def _send_email(self, msg: Message, recipients: List[str]) -> Tuple[str, List[str]]:
        """Just return msg string and recipients for testing."""
        return (msg.as_string(), recipients)


async def test_reload_notify(hass: HomeAssistant) -> None:
    """Verify we can reload the notify service."""
    with patch('homeassistant.components.smtp.notify.MailNotificationService.connection_is_valid'):
        assert await async_setup_component(
            hass,
            notify.DOMAIN,
            {
                notify.DOMAIN: [
                    {
                        'name': DOMAIN,
                        'platform': DOMAIN,
                        'recipient': 'test@example.com',
                        'sender': 'test@example.com'
                    }
                ]
            }
        )
        await hass.async_block_till_done()
    assert hass.services.has_service(notify.DOMAIN, DOMAIN)
    yaml_path: Path = get_fixture_path('configuration.yaml', 'smtp')
    with patch.object(hass_config, 'YAML_CONFIG_FILE', yaml_path), \
         patch('homeassistant.components.smtp.notify.MailNotificationService.connection_is_valid'):
        await hass.services.async_call(DOMAIN, SERVICE_RELOAD, {}, blocking=True)
        await hass.async_block_till_done()
    assert not hass.services.has_service(notify.DOMAIN, DOMAIN)
    assert hass.services.has_service(notify.DOMAIN, 'smtp_reloaded')


@pytest.fixture
def message() -> MockSMTP:
    """Return MockSMTP object with test data."""
    return MockSMTP(
        host='localhost',
        port=25,
        timeout=5,
        sender='test@test.com',
        retries=1,
        username='testuser',
        password='testpass',
        recipients=['recip1@example.com', 'testrecip@test.com'],
        subject='Home Assistant',
        debug=False,
        secure=True
    )


HTML: str = '\n        <!DOCTYPE html>\n        <html lang="en" xmlns="http://www.w3.org/1999/xhtml">\n            <head><meta charset="UTF-8"></head>\n            <body>\n              <div>\n                <h1>Intruder alert at apartment!!</h1>\n              </div>\n              <div>\n                <img alt="tests/testing_config/notify/test.jpg" src="cid:tests/testing_config/notify/test.jpg"/>\n              </div>\n            </body>\n        </html>'

EMAIL_DATA: List[Tuple[str, dict, str]] = [
    ('Test msg', {'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/mixed'),
    ('Test msg', {'html': HTML, 'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/related'),
    ('Test msg', {'html': HTML, 'images': ['tests/testing_config/notify/test_not_exists.jpg']}, 'Content-Type: multipart/related'),
    ('Test msg', {'html': HTML, 'images': ['tests/testing_config/notify/test.pdf']}, 'Content-Type: multipart/related')
]


@pytest.mark.parametrize(
    ('message_data', 'data', 'content_type'),
    EMAIL_DATA,
    ids=[
        'Tests when sending text message and images.',
        'Tests when sending text message, HTML Template and images.',
        'Tests when image does not exist at mentioned location.',
        'Tests when image type cannot be detected or is of wrong type.'
    ]
)
def test_send_message(
    hass: HomeAssistant,
    message_data: str,
    data: dict,
    content_type: str,
    message: MockSMTP
) -> None:
    """Verify if we can send messages of all types correctly."""
    sample_email: str = '<mock@mock>'
    message.hass = hass
    hass.config.allowlist_external_dirs.add(Path('tests/testing_config').resolve())
    with patch('email.utils.make_msgid', return_value=sample_email):
        result: str
        _, _ = message.send_message(message_data, data=data)
        assert content_type in result


@pytest.mark.parametrize(
    ('message_data', 'data', 'content_type'),
    [
        ('Test msg', {'images': ['tests/testing_config/notify/test.jpg']}, 'Content-Type: multipart/mixed')
    ]
)
def test_sending_insecure_files_fails(
    hass: HomeAssistant,
    message_data: str,
    data: dict,
    content_type: str,
    message: MockSMTP
) -> None:
    """Verify if we cannot send messages with insecure attachments."""
    sample_email: str = '<mock@mock>'
    message.hass = hass
    with patch('email.utils.make_msgid', return_value=sample_email), \
         pytest.raises(ServiceValidationError) as exc:
        _, _ = message.send_message(message_data, data=data)
    assert exc.value.translation_key == 'remote_path_not_allowed'
    assert exc.value.translation_domain == DOMAIN
    assert str(exc.value.translation_placeholders['file_path']) == 'tests/testing_config/notify'
    assert exc.value.translation_placeholders['url']
    assert exc.value.translation_placeholders['file_name'] == 'test.jpg'


def test_send_text_message(hass: HomeAssistant, message: MockSMTP) -> None:
    """Verify if we can send simple text message."""
    expected: str = '^Content-Type: text/plain; charset="us-ascii"\nMIME-Version: 1.0\nContent-Transfer-Encoding: 7bit\nSubject: Home Assistant\nTo: recip1@example.com,testrecip@test.com\nFrom: Home Assistant <test@test.com>\nX-Mailer: Home Assistant\nDate: [^\n]+\nMessage-Id: <[^@]+@[^>]+>\n\nTest msg$'
    sample_email: str = '<mock@mock>'
    message_data: str = 'Test msg'
    with patch('email.utils.make_msgid', return_value=sample_email):
        result: str
        _, _ = message.send_message(message_data)
        assert re.search(expected, result) is not None


@pytest.mark.parametrize(
    'target',
    [None, 'target@example.com'],
    ids=[
        'Verify we can send email to default recipient.',
        'Verify email recipient can be overwritten by target arg.'
    ]
)
def test_send_target_message(
    target: Optional[str],
    hass: HomeAssistant,
    message: MockSMTP
) -> None:
    """Verify if we can send email to correct recipient."""
    sample_email: str = '<mock@mock>'
    message_data: str = 'Test msg'
    with patch('email.utils.make_msgid', return_value=sample_email):
        if not target:
            expected_recipient: List[str] = ['recip1@example.com', 'testrecip@test.com']
        else:
            expected_recipient: str = target
        _, recipient = message.send_message(message_data, target=target)
        assert recipient == expected_recipient
