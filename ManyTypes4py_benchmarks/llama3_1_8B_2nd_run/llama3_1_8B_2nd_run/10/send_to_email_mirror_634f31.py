import base64
import email.parser
import email.policy
import os
from email.message import EmailMessage
from typing import Any, Optional
import orjson
from django.conf import settings
from django.core.management.base import CommandError, CommandParser
from typing_extensions import override
from zerver.lib.email_mirror import mirror_email_message
from zerver.lib.email_mirror_helpers import encode_email_address, get_channel_email_token
from zerver.lib.management import ZulipBaseCommand
from zerver.models import Realm, UserProfile
from zerver.models.realms import get_realm
from zerver.models.streams import get_stream
from zerver.models.users import get_system_bot, get_user_profile_by_email, get_user_profile_by_id

class Command(ZulipBaseCommand):
    help: str = '\nSend specified email from a fixture file to the email mirror\nExample:\n./manage.py send_to_email_mirror --fixture=zerver/tests/fixtures/emails/filename\n\n'

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('-f', '--fixture', help="The path to the email message you'd like to send to the email mirror.\nAccepted formats: json or raw email file. See zerver/tests/fixtures/email/ for examples")
        parser.add_argument('-s', '--stream', help="The name of the stream to which you'd like to send the message. Default: Denmark")
        parser.add_argument('--sender-id', type=int, help='The ID of a user or bot which should appear as the sender; Default: ID of Email gateway bot')
        self.add_realm_args(parser, help='Specify which realm to connect to; default is zulip')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if options['fixture'] is None:
            self.print_help('./manage.py', 'send_to_email_mirror')
            raise CommandError
        if options['stream'] is None:
            stream = 'Denmark'
        else:
            stream = options['stream']
        realm: Optional[Realm] = self.get_realm(options)
        if realm is None:
            realm = get_realm('zulip')
        email_gateway_bot: UserProfile = get_system_bot(settings.EMAIL_GATEWAY_BOT, realm.id)
        if options['sender_id'] is None:
            sender: UserProfile = email_gateway_bot
        else:
            sender = get_user_profile_by_id(options['sender_id'])
        full_fixture_path: str = os.path.join(settings.DEPLOY_ROOT, options['fixture'])
        message: EmailMessage = self._parse_email_fixture(full_fixture_path)
        creator: UserProfile = get_user_profile_by_email(message['From'])
        if sender.id not in [creator.id, email_gateway_bot.id] and sender.bot_owner_id != creator.id:
            raise CommandError("The sender ID must be either the current user's ID, the email gateway bot's ID, or the ID of a bot owned by the user.")
        self._prepare_message(message, realm, stream, creator, sender)
        mirror_email_message(message['To'].addresses[0].addr_spec, base64.b64encode(message.as_bytes()).decode())

    def _does_fixture_path_exist(self, fixture_path: str) -> bool:
        return os.path.exists(fixture_path)

    def _parse_email_json_fixture(self, fixture_path: str) -> EmailMessage:
        with open(fixture_path, 'rb') as fp:
            json_content = orjson.loads(fp.read())[0]
        message = EmailMessage()
        message['From'] = json_content['from']
        message['Subject'] = json_content['subject']
        message.set_content(json_content['body'])
        return message

    def _parse_email_fixture(self, fixture_path: str) -> EmailMessage:
        if not self._does_fixture_path_exist(fixture_path):
            raise CommandError(f'Fixture {fixture_path} does not exist')
        if fixture_path.endswith('.json'):
            return self._parse_email_json_fixture(fixture_path)
        else:
            with open(fixture_path, 'rb') as fp:
                return email.parser.BytesParser(_class=EmailMessage, policy=email.policy.default).parse(fp)

    def _prepare_message(self, message: EmailMessage, realm: Realm, stream_name: str, creator: UserProfile, sender: UserProfile) -> None:
        stream: 'Stream' = get_stream(stream_name, realm)
        email_token: str = get_channel_email_token(stream, creator=creator, sender=sender)
        recipient_headers: list[str] = ['X-Gm-Original-To', 'Delivered-To', 'Envelope-To', 'Resent-To', 'Resent-CC', 'CC']
        for header in recipient_headers:
            if header in message:
                del message[header]
                message[header] = encode_email_address(stream.name, email_token)
        if 'To' in message:
            del message['To']
        message['To'] = encode_email_address(stream.name, email_token)
