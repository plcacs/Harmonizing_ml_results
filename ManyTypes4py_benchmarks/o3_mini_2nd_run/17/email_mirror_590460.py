from typing import Optional, Tuple, Dict, Any
import logging
import re
import secrets
from email.headerregistry import Address, AddressHeader
from email.message import EmailMessage
from re import Match
from django.conf import settings
from django.utils.translation import gettext as _
from typing_extensions import override
from zerver.actions.message_send import check_send_message, internal_send_group_direct_message, internal_send_private_message, internal_send_stream_message
from zerver.lib.display_recipient import get_display_recipient
from zerver.lib.email_mirror_helpers import ZulipEmailForwardError, ZulipEmailForwardUserError, decode_email_address, get_email_gateway_message_string_from_address
from zerver.lib.email_notifications import convert_html_to_markdown
from zerver.lib.exceptions import JsonableError, RateLimitedError
from zerver.lib.message import normalize_body, truncate_content, truncate_topic
from zerver.lib.queue import queue_json_publish_rollback_unsafe
from zerver.lib.rate_limiter import RateLimitedObject
from zerver.lib.send_email import FromAddress
from zerver.lib.streams import access_stream_for_send_message
from zerver.lib.string_validation import is_character_printable
from zerver.lib.upload import upload_message_attachment
from zerver.models import ChannelEmailAddress, Message, MissedMessageEmailAddress, Realm, Recipient, Stream, UserProfile
from zerver.models.clients import get_client
from zerver.models.streams import get_stream_by_id_in_realm
from zerver.models.users import get_system_bot, get_user_profile_by_id
from zproject.backends import is_user_active

logger = logging.getLogger(__name__)

def redact_email_address(error_message: str) -> str:
    if not settings.EMAIL_GATEWAY_EXTRA_PATTERN_HACK:
        domain = Address(addr_spec=settings.EMAIL_GATEWAY_PATTERN).domain
    else:
        domain = settings.EMAIL_GATEWAY_EXTRA_PATTERN_HACK.removeprefix('@')

    def redact(address_match: Match[str]) -> str:
        email_address = address_match[0]
        if is_missed_message_address(email_address):
            annotation = ' <Missed message address>'
        else:
            try:
                target_stream_id = decode_stream_email_address(email_address)[0].channel_id
                annotation = f' <Address to stream id: {target_stream_id}>'
            except ZulipEmailForwardError:
                annotation = ' <Invalid address>'
        return 'X' * len(address_match[1]) + address_match[2] + annotation
    return re.sub(f'\\b(\\S*?)(@{re.escape(domain)})', redact, error_message)

def log_error(email_message: EmailMessage, error_message: str, to: Optional[str]) -> None:
    recipient = to or 'No recipient found'
    error_message = 'Sender: {}\nTo: {}\n{}'.format(email_message.get('From'), recipient, error_message)
    error_message = redact_email_address(error_message)
    logger.error(error_message)

def generate_missed_message_token() -> str:
    return 'mm' + secrets.token_hex(16)

def is_missed_message_address(address: str) -> bool:
    try:
        msg_string = get_email_gateway_message_string_from_address(address)
    except ZulipEmailForwardError:
        return False
    return is_mm_32_format(msg_string)

def is_mm_32_format(msg_string: str) -> bool:
    """
    Missed message strings are formatted with a little "mm" prefix
    followed by a randomly generated 32-character string.
    """
    return msg_string is not None and msg_string.startswith('mm') and (len(msg_string) == 34)

def get_missed_message_token_from_address(address: str) -> str:
    msg_string = get_email_gateway_message_string_from_address(address)
    if not is_mm_32_format(msg_string):
        raise ZulipEmailForwardError('Could not parse missed message address')
    return msg_string

def get_usable_missed_message_address(address: str) -> MissedMessageEmailAddress:
    token = get_missed_message_token_from_address(address)
    try:
        mm_address = MissedMessageEmailAddress.objects.select_related(
            'user_profile', 'user_profile__realm', 'user_profile__realm__can_access_all_users_group',
            'user_profile__realm__can_access_all_users_group__named_user_group',
            'user_profile__realm__direct_message_initiator_group',
            'user_profile__realm__direct_message_initiator_group__named_user_group',
            'user_profile__realm__direct_message_permission_group',
            'user_profile__realm__direct_message_permission_group__named_user_group',
            'message', 'message__sender', 'message__recipient', 'message__sender__recipient'
        ).get(email_token=token)
    except MissedMessageEmailAddress.DoesNotExist:
        raise ZulipEmailForwardError('Zulip notification reply address is invalid.')
    return mm_address

def create_missed_message_address(user_profile: UserProfile, message: Message) -> str:
    if settings.EMAIL_GATEWAY_PATTERN == '':
        return FromAddress.NOREPLY
    mm_address = MissedMessageEmailAddress.objects.create(message=message, user_profile=user_profile, email_token=generate_missed_message_token())
    return str(mm_address)

def construct_zulip_body(message: EmailMessage, realm: Realm, *, sender: UserProfile, show_sender: bool = False, include_quotes: bool = False, include_footer: bool = False, prefer_text: bool = True) -> str:
    body = extract_body(message, include_quotes, prefer_text)
    body = body.replace('\x00', '')
    if not include_footer:
        body = filter_footer(body)
    if not body.endswith('\n'):
        body += '\n'
    if not body.rstrip():
        body = '(No email body)'
    preamble = ''
    if show_sender:
        from_address = str(message.get('From', ''))
        preamble = f'From: {from_address}\n'
    postamble = extract_and_upload_attachments(message, realm, sender)
    if postamble != '':
        postamble = '\n' + postamble
    body = truncate_content(body, settings.MAX_MESSAGE_LENGTH - len(preamble) - len(postamble), '\n[message truncated]')
    return preamble + body + postamble

def send_zulip(sender: UserProfile, stream: Stream, topic_name: str, content: str) -> None:
    internal_send_stream_message(sender, stream, truncate_topic(topic_name), normalize_body(content), email_gateway=True)

def send_mm_reply_to_stream(user_profile: UserProfile, stream: Stream, topic_name: str, body: str) -> None:
    try:
        check_send_message(sender=user_profile, client=get_client('Internal'), recipient_type_name='stream', message_to=[stream.id], topic_name=topic_name, message_content=body)
    except JsonableError as error:
        error_message = _('Error sending message to channel {channel_name} via message notification email reply:\n{error_message}').format(channel_name=stream.name, error_message=error.msg)
        internal_send_private_message(get_system_bot(settings.NOTIFICATION_BOT, user_profile.realm_id), user_profile, error_message)

def get_message_part_by_type(message: EmailMessage, content_type: str) -> Optional[str]:
    charsets = message.get_charsets()
    for idx, part in enumerate(message.walk()):
        if part.get_content_type() == content_type:
            content = part.get_payload(decode=True)
            assert isinstance(content, bytes)
            charset = charsets[idx]
            if charset is not None:
                try:
                    return content.decode(charset, errors='ignore')
                except LookupError:
                    pass
            return content.decode('us-ascii', errors='ignore')
    return None

def extract_body(message: EmailMessage, include_quotes: bool = False, prefer_text: bool = True) -> str:
    plaintext_content = extract_plaintext_body(message, include_quotes)
    html_content = extract_html_body(message, include_quotes)
    if plaintext_content is None and html_content is None:
        logger.warning('Content types: %s', [part.get_content_type() for part in message.walk()])
        raise ZulipEmailForwardUserError('Unable to find plaintext or HTML message body')
    if not plaintext_content and (not html_content):
        raise ZulipEmailForwardUserError('Email has no nonempty body sections; ignoring.')
    if prefer_text:
        if plaintext_content:
            return plaintext_content
        else:
            assert html_content
            return html_content
    elif html_content:
        return html_content
    else:
        assert plaintext_content
        return plaintext_content

talon_initialized = False

def extract_plaintext_body(message: EmailMessage, include_quotes: bool = False) -> Optional[str]:
    import talon_core
    global talon_initialized
    if not talon_initialized:
        talon_core.init()
        talon_initialized = True
    plaintext_content = get_message_part_by_type(message, 'text/plain')
    if plaintext_content is not None:
        if include_quotes:
            return plaintext_content
        else:
            return talon_core.quotations.extract_from_plain(plaintext_content)
    else:
        return None

def extract_html_body(message: EmailMessage, include_quotes: bool = False) -> Optional[str]:
    import talon_core
    global talon_initialized
    if not talon_initialized:
        talon_core.init()
        talon_initialized = True
    html_content = get_message_part_by_type(message, 'text/html')
    if html_content is not None:
        if include_quotes:
            return convert_html_to_markdown(html_content)
        else:
            return convert_html_to_markdown(talon_core.quotations.extract_from_html(html_content))
    else:
        return None

def filter_footer(text: str) -> str:
    possible_footers = [line for line in text.split('\n') if line.strip() == '--']
    if len(possible_footers) != 1:
        return text
    return re.split('^\\s*--\\s*$', text, maxsplit=1, flags=re.MULTILINE)[0].strip()

def extract_and_upload_attachments(message: EmailMessage, realm: Realm, sender: UserProfile) -> str:
    attachment_links = []
    for part in message.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()
        if filename:
            attachment = part.get_payload(decode=True)
            if isinstance(attachment, bytes):
                upload_url, filename = upload_message_attachment(filename, content_type, attachment, sender, target_realm=realm)
                filename = re.sub('\\[|\\]', '', filename)
                formatted_link = f'[{filename}]({upload_url})'
                attachment_links.append(formatted_link)
            else:
                logger.warning('Payload is not bytes (invalid attachment %s in message from %s).', filename, message.get('From'))
    return '\n'.join(attachment_links)

def decode_stream_email_address(email: str) -> Tuple[ChannelEmailAddress, Dict[str, Any]]:
    token, options = decode_email_address(email)
    try:
        channel_email_address = ChannelEmailAddress.objects.select_related('channel', 'sender', 'creator', 'realm').get(email_token=token)
    except ChannelEmailAddress.DoesNotExist:
        raise ZulipEmailForwardError('Bad stream token from email recipient ' + email)
    return (channel_email_address, options)

def find_emailgateway_recipient(message: EmailMessage) -> str:
    recipient_headers = ['X-Gm-Original-To', 'Delivered-To', 'Envelope-To', 'Resent-To', 'Resent-CC', 'To', 'CC']
    pattern_parts = [re.escape(part) for part in settings.EMAIL_GATEWAY_PATTERN.split('%s')]
    match_email_re = re.compile('.*?'.join(pattern_parts))
    for header_name in recipient_headers:
        for header_value in message.get_all(header_name, []):
            if isinstance(header_value, AddressHeader):
                emails = [addr.addr_spec for addr in header_value.addresses]
            else:
                emails = [str(header_value)]
            for email in emails:
                if match_email_re.match(email):
                    return email
    raise ZulipEmailForwardError('Missing recipient in mirror email')

def strip_from_subject(subject: str) -> str:
    reg = '([\\[\\(] *)?\\b(RE|AW|FWD?) *([-:;)\\]][ :;\\])-]*|$)|\\]+ *$'
    stripped = re.sub(reg, '', subject, flags=re.IGNORECASE | re.MULTILINE)
    return stripped.strip()

def is_forwarded(subject: str) -> bool:
    reg = '([\\[\\(] *)?\\b(FWD?) *([-:;)\\]][ :;\\])-]*|$)|\\]+ *$'
    return bool(re.match(reg, subject, flags=re.IGNORECASE))

def process_stream_message(to: str, message: EmailMessage) -> None:
    subject_header = message.get('Subject', '')
    subject = strip_from_subject(subject_header)
    subject = ''.join([char for char in subject if is_character_printable(char)])
    subject = subject or _('Email with no subject')
    channel_email_address, options = decode_stream_email_address(to)
    if 'include_quotes' not in options:
        options['include_quotes'] = is_forwarded(subject_header)
    channel = channel_email_address.channel
    sender = channel_email_address.sender
    creator = channel_email_address.creator
    realm = channel_email_address.realm
    email_gateway_bot = get_system_bot(settings.EMAIL_GATEWAY_BOT, realm.id)
    if sender.id == email_gateway_bot.id and creator is not None:
        user_for_access_check = creator
    else:
        user_for_access_check = sender
    try:
        access_stream_for_send_message(user_for_access_check, channel, forwarder_user_profile=None)
    except JsonableError as e:
        logger.info('Failed to process email to %s (%s): %s', channel.name, realm.string_id, e)
        return
    body = construct_zulip_body(message, realm, sender=sender, **options)
    send_zulip(sender, channel, subject, body)
    logger.info('Successfully processed email to %s (%s)', channel.name, realm.string_id)

def process_missed_message(to: str, message: EmailMessage) -> None:
    mm_address = get_usable_missed_message_address(to)
    mm_address.increment_times_used()
    user_profile = mm_address.user_profile
    topic_name = mm_address.message.topic_name()
    if mm_address.message.recipient.type == Recipient.PERSONAL:
        recipient = mm_address.message.sender.recipient
    else:
        recipient = mm_address.message.recipient
    if not is_user_active(user_profile):
        logger.warning('Sending user is not active. Ignoring this message notification email.')
        return
    body = construct_zulip_body(message, user_profile.realm, sender=user_profile)
    assert recipient is not None
    if recipient.type == Recipient.STREAM:
        stream = get_stream_by_id_in_realm(recipient.type_id, user_profile.realm)
        send_mm_reply_to_stream(user_profile, stream, topic_name, body)
        recipient_str = stream.name
    elif recipient.type == Recipient.PERSONAL:
        recipient_user_id = recipient.type_id
        recipient_user = get_user_profile_by_id(recipient_user_id)
        recipient_str = recipient_user.email
        internal_send_private_message(user_profile, recipient_user, body)
    elif recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        display_recipient = get_display_recipient(recipient)
        emails = [user_dict['email'] for user_dict in display_recipient]
        recipient_str = ', '.join(emails)
        internal_send_group_direct_message(user_profile.realm, user_profile, body, emails=emails)
    else:
        raise AssertionError('Invalid recipient type!')
    logger.info('Successfully processed email from user %s to %s', user_profile.id, recipient_str)

def process_message(message: EmailMessage, rcpt_to: Optional[str] = None) -> None:
    to: Optional[str] = None
    try:
        if rcpt_to is not None:
            to = rcpt_to
        else:
            to = find_emailgateway_recipient(message)
        if is_missed_message_address(to):
            process_missed_message(to, message)
        else:
            process_stream_message(to, message)
    except ZulipEmailForwardUserError as e:
        logger.info(e.args[0])
    except ZulipEmailForwardError as e:
        log_error(message, e.args[0], to)

def validate_to_address(rcpt_to: str) -> None:
    if is_missed_message_address(rcpt_to):
        get_usable_missed_message_address(rcpt_to)
    else:
        decode_stream_email_address(rcpt_to)

def mirror_email_message(rcpt_to: str, msg_base64: str) -> Dict[str, str]:
    try:
        validate_to_address(rcpt_to)
    except ZulipEmailForwardError as e:
        return {'status': 'error', 'msg': f'5.1.1 Bad destination mailbox address: {e}'}
    queue_json_publish_rollback_unsafe('email_mirror', {'rcpt_to': rcpt_to, 'msg_base64': msg_base64})
    return {'status': 'success'}

class RateLimitedRealmMirror(RateLimitedObject):
    realm: Realm

    def __init__(self, realm: Realm) -> None:
        self.realm = realm
        super().__init__()

    @override
    def key(self) -> str:
        return f'{type(self).__name__}:{self.realm.string_id}'

    @override
    def rules(self) -> Any:
        return settings.RATE_LIMITING_MIRROR_REALM_RULES

def rate_limit_mirror_by_realm(recipient_realm: Realm) -> None:
    ratelimited, secs_to_freedom = RateLimitedRealmMirror(recipient_realm).rate_limit()
    if ratelimited:
        raise RateLimitedError(secs_to_freedom)