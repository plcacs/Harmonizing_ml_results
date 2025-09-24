import hashlib
import logging
import os
import smtplib
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import timedelta
from email.headerregistry import Address
from email.message import EmailMessage
from email.parser import Parser
from email.policy import default
from email.utils import formataddr, parseaddr
from typing import Any, Optional, Union, List, Dict, Set, Tuple, cast
import backoff
import css_inline
import orjson
from django.conf import settings
from django.core.mail import EmailMultiAlternatives, get_connection
from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail.backends.smtp import EmailBackend
from django.core.mail.message import sanitize_address
from django.core.management import CommandError
from django.db import transaction
from django.db.models import QuerySet
from django.db.models.functions import Lower
from django.http import HttpRequest
from django.template import loader
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from confirmation.models import generate_key
from zerver.lib.logging_util import log_to_file
from zerver.models import Realm, ScheduledEmail, UserProfile
from zerver.models.scheduled_jobs import EMAIL_TYPES
from zerver.models.users import get_user_profile_by_id
from zproject.email_backends import EmailLogBackEnd, get_forward_address
if settings.ZILENCER_ENABLED:
    from zilencer.models import RemoteZulipServer
MAX_CONNECTION_TRIES = 3
logger = logging.getLogger('zulip.send_email')
log_to_file(logger, settings.EMAIL_LOG_PATH)

def get_inliner_instance() -> css_inline.CSSInliner:
    return css_inline.CSSInliner()

class FromAddress:
    SUPPORT: str = parseaddr(settings.ZULIP_ADMINISTRATOR)[1]
    NOREPLY: str = parseaddr(settings.NOREPLY_EMAIL_ADDRESS)[1]
    support_placeholder: str = 'SUPPORT'
    no_reply_placeholder: str = 'NO_REPLY'
    tokenized_no_reply_placeholder: str = 'TOKENIZED_NO_REPLY'

    @staticmethod
    def tokenized_no_reply_address() -> str:
        if settings.ADD_TOKENS_TO_NOREPLY_ADDRESS:
            return parseaddr(settings.TOKENIZED_NOREPLY_EMAIL_ADDRESS)[1].format(token=generate_key())
        return FromAddress.NOREPLY

    @staticmethod
    def security_email_from_name(language: Optional[str] = None, user_profile: Optional[UserProfile] = None) -> str:
        if language is None:
            assert user_profile is not None
            language = user_profile.default_language
        with override_language(language):
            return _('{service_name} account security').format(service_name=settings.INSTALLATION_NAME)

def build_email(template_prefix: str, to_user_ids: Optional[List[int]] = None, to_emails: Optional[List[str]] = None, from_name: Optional[str] = None, from_address: Optional[str] = None, reply_to_email: Optional[str] = None, language: Optional[str] = None, context: Dict[str, Any] = {}, realm: Optional[Realm] = None) -> EmailMultiAlternatives:
    assert (to_user_ids is None) ^ (to_emails is None)
    if to_user_ids is not None:
        to_users = [get_user_profile_by_id(to_user_id) for to_user_id in to_user_ids]
        if realm is None:
            assert len({to_user.realm_id for to_user in to_users}) == 1
            realm = to_users[0].realm
        to_emails = []
        for to_user in to_users:
            stringified = str(Address(display_name=to_user.full_name, addr_spec=to_user.delivery_email))
            if len(sanitize_address(stringified, 'utf-8')) > 320:
                stringified = str(Address(addr_spec=to_user.delivery_email))
            to_emails.append(stringified)
    extra_headers: Dict[str, str] = {'X-Auto-Response-Suppress': 'All'}
    if realm is not None:
        extra_headers['List-Id'] = formataddr((realm.name, realm.host))
    assert settings.STATIC_URL is not None
    context = {**context, 'support_email': FromAddress.SUPPORT, 'email_images_base_url': settings.STATIC_URL + 'images/emails', 'physical_address': settings.PHYSICAL_ADDRESS}

    def get_inlined_template(template: str) -> str:
        inliner = get_inliner_instance()
        return inliner.inline(template)

    def render_templates() -> Tuple[str, str, str]:
        email_subject = loader.render_to_string(template_prefix + '.subject.txt', context=context, using='Jinja2_plaintext').strip().replace('\n', '')
        message = loader.render_to_string(template_prefix + '.txt', context=context, using='Jinja2_plaintext')
        html_message = loader.render_to_string(template_prefix + '.html', context)
        return (get_inlined_template(html_message), message, email_subject)
    if not language and to_user_ids is not None:
        language = to_users[0].default_language
    if language:
        with override_language(language):
            (html_message, message, email_subject) = render_templates()
    else:
        (html_message, message, email_subject) = render_templates()
        logger.warning("Missing language for email template '%s'", template_prefix)
    if from_name is None:
        from_name = 'Zulip'
    if from_address is None:
        from_address = FromAddress.NOREPLY
    if from_address == FromAddress.tokenized_no_reply_placeholder:
        from_address = FromAddress.tokenized_no_reply_address()
    if from_address == FromAddress.no_reply_placeholder:
        from_address = FromAddress.NOREPLY
    if from_address == FromAddress.support_placeholder:
        from_address = FromAddress.SUPPORT
    extra_headers['From'] = str(Address(display_name=from_name, addr_spec=from_address))
    if len(sanitize_address(extra_headers['From'], 'utf-8')) > 320:
        extra_headers['From'] = str(Address(addr_spec=from_address))
    if 'unsubscribe_link' in context:
        extra_headers['List-Unsubscribe'] = f"<{context['unsubscribe_link']}>"
        if not context.get('remote_server_email', False):
            extra_headers['List-Unsubscribe-Post'] = 'List-Unsubscribe=One-Click'
    reply_to = None
    if reply_to_email is not None:
        reply_to = [reply_to_email]
    elif from_address == FromAddress.NOREPLY:
        reply_to = [FromAddress.NOREPLY]
    envelope_from = FromAddress.NOREPLY
    mail = EmailMultiAlternatives(email_subject, message, envelope_from, to_emails, reply_to=reply_to, headers=extra_headers)
    if html_message is not None:
        mail.attach_alternative(html_message, 'text/html')
    return mail

class EmailNotDeliveredError(Exception):
    pass

class DoubledEmailArgumentError(CommandError):

    def __init__(self, argument_name: str) -> None:
        msg = f"Argument '{argument_name}' is ambiguously present in both options and email template."
        super().__init__(msg)

class NoEmailArgumentError(CommandError):

    def __init__(self, argument_name: str) -> None:
        msg = f"Argument '{argument_name}' is required in either options or email template."
        super().__init__(msg)

def send_email(template_prefix: str, to_user_ids: Optional[List[int]] = None, to_emails: Optional[List[str]] = None, from_name: Optional[str] = None, from_address: Optional[str] = None, reply_to_email: Optional[str] = None, language: Optional[str] = None, context: Dict[str, Any] = {}, realm: Optional[Realm] = None, connection: Optional[BaseEmailBackend] = None, dry_run: bool = False, request: Optional[HttpRequest] = None) -> None:
    mail = build_email(template_prefix, to_user_ids=to_user_ids, to_emails=to_emails, from_name=from_name, from_address=from_address, reply_to_email=reply_to_email, language=language, context=context, realm=realm)
    template = template_prefix.split('/')[-1]
    log_email_config_errors()
    if dry_run:
        print(mail.message().get_payload()[0])
        return
    if connection is None:
        connection = get_connection()
    cause = ''
    if request is not None:
        cause = f" (triggered from {request.META['REMOTE_ADDR']})"
    logging_recipient: Union[str, List[str]] = mail.to
    if realm is not None:
        logging_recipient = f'{mail.to} in {realm.string_id}'
    logger.info('Sending %s email to %s%s', template, logging_recipient, cause)
    try:
        if connection.send_messages([mail]) == 0:
            logger.error('Unknown error sending %s email to %s', template, mail.to)
            raise EmailNotDeliveredError
    except smtplib.SMTPResponseException as e:
        logger.exception('Error sending %s email to %s with error code %s: %s', template, mail.to, e.smtp_code, e.smtp_error, stack_info=True)
        raise EmailNotDeliveredError
    except smtplib.SMTPException as e:
        logger.exception('Error sending %s email to %s: %s', template, mail.to, e, stack_info=True)
        raise EmailNotDeliveredError

@backoff.on_exception(backoff.expo, OSError, max_tries=MAX_CONNECTION_TRIES, logger=None)
def initialize_connection(connection: Optional[BaseEmailBackend] = None) -> BaseEmailBackend:
    if not connection:
        connection = get_connection()
        assert connection is not None
    if connection.open():
        return connection
    if isinstance(connection, EmailLogBackEnd) and (not get_forward_address()):
        return connection
    if isinstance(connection, EmailBackend):
        try:
            assert connection.connection is not None
            status = connection.connection.noop()[0]
        except Exception:
            status = -1
        if status != 250:
            connection.close()
            connection.open()
    return connection

def send_future_email(template_prefix: str, realm: Realm, to_user_ids: Optional[List[int]] = None, to_emails: Optional[List[str]] = None, from_name: Optional[str] = None, from_address: Optional[str] = None, language: Optional[str] = None, context: Mapping[str, Any] = {}, delay: timedelta = timedelta(0)) -> None:
    template_name = template_prefix.split('/')[-1]
    email_fields = {'template_prefix': template_prefix, 'from_name': from_name, 'from_address': from_address, 'language': language, 'context': context}
    if settings.DEVELOPMENT_LOG_EMAILS:
        send_email(template_prefix, to_user_ids=to_user_ids, to_emails=to_emails, from_name=from_name, from_address=from_address, language=language, context=context)
    assert (to_user_ids is None) ^ (to_emails is None)
    with transaction.atomic(savepoint=False):
        email = ScheduledEmail.objects.create(type=EMAIL_TYPES[template_name], scheduled_timestamp=timezone_now() + delay, realm=realm, data=orjson.dumps(email_fields).decode())
        try:
            if to_user_ids is not None:
                email.users.add(*to_user_ids)
            else:
                assert to_emails is not None
                assert len(to_emails) == 1
                email.address = parseaddr(to_emails[0])[1]
                email.save()
        except Exception as e:
            email.delete()
            raise e

def send_email_to_admins(template_prefix: str, realm: Realm, from_name: Optional[str] = None, from_address: Optional[str] = None, language: Optional[str] = None, context: Mapping[str, Any] = {}) -> None:
    admins = realm.get_human_admin_users()
    admin_user_ids = [admin.id for admin in admins]
    send_email(template_prefix, to_user_ids=admin_user_ids, from_name=from_name, from_address=from_address, language=language, context=context)

def send_email_to_billing_admins_and_realm_owners(template_prefix: str, realm: Realm, from_name: Optional[str] = None, from_address: Optional[str] = None, language: Optional[str] = None, context: Mapping[str, Any] = {}) -> None:
    send_email(template_prefix, to_user_ids=[user.id for user in realm.get_human_billing_admin_and_realm_owner_users()], from_name=from_name, from_address=from_address, language=language, context=context)

def clear_scheduled_invitation_emails(email: str) -> None:
    """Unlike most scheduled emails, invitation emails don't have an
    existing user object to key off of, so we filter by address here."""
    items = ScheduledEmail.objects.filter(address__iexact=email, type=ScheduledEmail.INVITATION_REMINDER)
    items.delete()

@transaction.atomic(savepoint=False)
def clear_scheduled_emails(user_id: int, email_type: Optional[int] = None) -> None:
    items = ScheduledEmail.objects.filter(users__in=[user_id]).prefetch_related('users').select_for_update()
    if email_type is not None:
        items = items.filter(type=email_type)
    for item in items:
        item.users.remove(user_id)
        if not item.users.all().exists():
            item.delete()

def handle_send_email_format_changes(job: Dict[str, Any]) -> None:
    if 'to_email' in job:
        if job['to_email'] is not None:
            job['to_emails'] = [job['to_email']]
        del job['to_email']
    if 'to_user_id' in job:
        if job['to_user_id'] is not None:
            job['to_user_ids'] = [job['to_user_id']]
        del job['to_user_id']

def deliver_scheduled_emails(email: ScheduledEmail) -> None:
    data = orjson.loads(email.data)
    user_ids = list(email.users.values_list('id', flat=True))
    if not user_ids and (not email.address):
        logger.warning('ScheduledEmail %s at %s had empty users and address attributes: %r', email.id, email.scheduled_timestamp, data)
        email.delete()
        return
    if user_ids:
        data['to_user_ids'] = user_ids
    if email.address is not None:
        data['to_emails'] = [email.address]
    handle_send_email_format_changes(data)
    send_email(**data)
    email.delete()

def get_header(option: Optional[str], header: Optional[str], name: str) -> str:
    if option and header:
        raise DoubledEmailArgumentError(name)
    if not option and (not header):
        raise NoEmailArgumentError(name)
    return str(option or header)

def custom_email_sender(markdown_template_path: str, dry_run: bool, subject: Optional[str] = None, from_address: str = FromAddress.SUPPORT, from_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs: Any) -> Callable[..., None]:
    with open(markdown_template_path) as f:
        text = f.read()
        parsed_email_template = Parser(_class=EmailMessage, policy=default).parsestr(text)
        email_template_hash = hashlib.sha256(text.encode()).hexdigest()[0:32]
    email_id = f'zerver/emails/custom/custom_email_{email_template_hash}'
    markdown_email_base_template_path = 'templates/zerver/emails/custom_email_base.pre.html'
    html_template_path = f'templates/{email_id}.html'
    plain_text_template_path = f'templates/{email_id}.txt'
    subject_path = f'templates/{email_id}.subject.txt'
    os.makedirs(os.path.dirname(html_template_path), exist_ok=True)
    with open(plain_text_template_path, 'w') as f:
        payload = parsed_email_template.get_payload()
        assert isinstance(payload, str)
        f.write(payload)
    from zerver.lib.templates import render_markdown_path
    rendered_input = render_markdown_path(plain_text_template_path.replace('templates/', ''))
    with open(html_template_path, 'w') as f, open(markdown_email_base_template_path) as base_template:
        f.write(base_template.read().replace('{{ rendered_input }}', rendered_input))
    manage_preferences_block_template_path = 'templates/zerver/emails/custom_email_base.pre.manage_preferences_block.txt'
    with open(plain_text_template_path, 'a') as f, open(manage_preferences_block_template_path) as manage_preferences_block:
        f.write(manage_preferences_block.read())
    with open(subject_path, 'w') as f:
        f.write(get_header(subject, parsed_email_template.get('subject'), 'subject'))

    def send_one_email(context: Dict[str, Any], to_user_id: Optional[int] = None, to_email: Optional[str] = None) -> None:
        assert to_user_id is not None or to_email is not None
        with suppress(EmailNotDeliveredError):
            send_email(email_id, to_user_ids=[to_user_id] if to_user_id is not None else None, to_emails=[to_email] if to_email is not None else None, from_address=from_address, reply_to_email=reply_to, from_name=get_header(from_name, parsed_email_template.get('from'), 'from_name'), context=context, dry_run=dry_run)
    return send_one_email

def send_custom_email(users: QuerySet[UserProfile], *, dry_run: bool, options: Dict[str, str],