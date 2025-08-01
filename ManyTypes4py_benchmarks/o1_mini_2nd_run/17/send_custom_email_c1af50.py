from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any, Dict, Optional
import orjson
from django.conf import settings
from django.db.models import Q, QuerySet
from typing_extensions import override
from confirmation.models import one_click_unsubscribe_link
from zerver.lib.management import ZulipBaseCommand
from zerver.lib.send_email import send_custom_email, send_custom_server_email
from zerver.models import Realm, UserProfile
if settings.ZILENCER_ENABLED:
    from zilencer.models import RemoteZulipServer


class Command(ZulipBaseCommand):
    help = '\n    Send a custom email with Zulip branding to the specified users.\n\n    Useful to send a notice to all users of a realm or server.\n\n    The From and Subject headers can be provided in the body of the Markdown\n    document used to generate the email, or on the command line.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        targets = parser.add_mutually_exclusive_group(required=True)
        targets.add_argument(
            '--entire-server',
            action='store_true',
            help='Send to every user on the server.'
        )
        targets.add_argument(
            '--marketing',
            action='store_true',
            help='Send to active users and realm owners with the enable_marketing_emails setting enabled.'
        )
        targets.add_argument(
            '--remote-servers',
            action='store_true',
            help='Send to registered contact email addresses for remote Zulip servers.'
        )
        targets.add_argument(
            '--all-sponsored-org-admins',
            action='store_true',
            help='Send to all organization administrators of sponsored organizations.'
        )
        self.add_user_list_args(
            targets,
            help='Email addresses of user(s) to send emails to.',
            all_users_help='Send to every user on the realm.'
        )
        self.add_realm_args(parser)
        parser.add_argument(
            '--json-file',
            help='Load the JSON file, and send to the users whose ids are the keys in that dict; the context for each email will be extended by each value in the dict.'
        )
        parser.add_argument(
            '--admins-only',
            help='Filter recipients selected via other options to only organization administrators',
            action='store_true'
        )
        parser.add_argument(
            '--markdown-template-path',
            '--path',
            required=True,
            help='Path to a Markdown-format body for the email.'
        )
        parser.add_argument(
            '--subject',
            help='Subject for the email. It can be declared in Markdown file in headers'
        )
        parser.add_argument(
            '--from-name',
            help='From line for the email. It can be declared in Markdown file in headers'
        )
        parser.add_argument(
            '--from-address',
            help='From email address'
        )
        parser.add_argument(
            '--reply-to',
            help='Optional reply-to line for the email'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Prints emails of the recipients and text of the email.'
        )

    @override
    def handle(
        self,
        *args: Any,
        dry_run: bool = False,
        admins_only: bool = False,
        **options: Any
    ) -> None:
        users: QuerySet[UserProfile] = UserProfile.objects.none()
        add_context: Optional[Callable[[Dict[str, Any], Any], None]] = None
        distinct_email: bool = False
        if options['remote_servers']:
            servers: QuerySet[RemoteZulipServer] = RemoteZulipServer.objects.filter(deactivated=False)
            add_server_context: Optional[Callable[[Dict[str, Any], RemoteZulipServer], None]] = None
            if options['json_file']:
                with open(options['json_file'], 'rb') as f:
                    server_data: Dict[str, Any] = orjson.loads(f.read())
                servers = RemoteZulipServer.objects.filter(
                    id__in=[int(server_id) for server_id in server_data]
                )

                def add_server_context_from_dict(context: Dict[str, Any], server: RemoteZulipServer) -> None:
                    context.update(server_data.get(str(server.id), {}))

                add_server_context = add_server_context_from_dict
            send_custom_server_email(
                servers,
                dry_run=dry_run,
                options=options,
                add_context=add_server_context
            )
            if dry_run:
                print('Would send the above email to:')
                for server in servers:
                    print(f'  {server.contact_email} ({server.hostname})')
            return
        if options['entire_server']:
            users = UserProfile.objects.filter(
                is_active=True,
                is_bot=False,
                is_mirror_dummy=False,
                realm__deactivated=False
            )
        elif options['marketing']:
            users = UserProfile.objects.filter(
                is_active=True,
                is_bot=False,
                is_mirror_dummy=False,
                realm__deactivated=False,
                enable_marketing_emails=True
            ).filter(
                Q(long_term_idle=False) |
                Q(role__in=[UserProfile.ROLE_REALM_OWNER, UserProfile.ROLE_REALM_ADMINISTRATOR])
            )
            distinct_email = True

            def add_marketing_unsubscribe(context: Dict[str, Any], user: UserProfile) -> None:
                context['unsubscribe_link'] = one_click_unsubscribe_link(user, 'marketing')

            add_context = add_marketing_unsubscribe
        elif options['all_sponsored_org_admins']:
            sponsored_realms: QuerySet[Realm] = Realm.objects.filter(
                plan_type=Realm.PLAN_TYPE_STANDARD_FREE,
                deactivated=False
            )
            admin_roles: list[int] = [
                UserProfile.ROLE_REALM_ADMINISTRATOR,
                UserProfile.ROLE_REALM_OWNER
            ]
            users = UserProfile.objects.filter(
                is_active=True,
                is_bot=False,
                is_mirror_dummy=False,
                role__in=admin_roles,
                realm__deactivated=False,
                realm__in=sponsored_realms
            )
            distinct_email = True
        else:
            realm: Realm = self.get_realm(options)
            users = self.get_users(options, realm, is_bot=False)
        if options['json_file']:
            with open(options['json_file'], 'rb') as f:
                user_data: Dict[str, Any] = orjson.loads(f.read())
            users = users.filter(id__in=[int(user_id) for user_id in user_data])

            def add_context_from_dict(context: Dict[str, Any], user: UserProfile) -> None:
                context.update(user_data.get(str(user.id), {}))

            add_context = add_context_from_dict
        if admins_only:
            users = users.filter(
                role__in=[
                    UserProfile.ROLE_REALM_ADMINISTRATOR,
                    UserProfile.ROLE_REALM_OWNER
                ]
            )
        if settings.TERMS_OF_SERVICE_VERSION is not None:
            users = users.exclude(
                Q(tos_version=None) |
                Q(tos_version=UserProfile.TOS_VERSION_BEFORE_FIRST_LOGIN)
            )
        users = send_custom_email(
            users,
            dry_run=dry_run,
            options=options,
            add_context=add_context,
            distinct_email=distinct_email
        )
        if dry_run:
            print('Would send the above email to:')
            for user in users:
                print(f'  {user.delivery_email} ({user.realm.string_id})')
