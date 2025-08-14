#!/usr/bin/env python3
import logging
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction, RawTextHelpFormatter, _ActionsContainer
from dataclasses import dataclass
from functools import reduce, wraps
from typing import Any, Optional, Callable, TypeVar

from django.conf import settings
from django.core import validators
from django.core.exceptions import MultipleObjectsReturned, ValidationError
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db.models import Q, QuerySet
from typing_extensions import override, Protocol

from zerver.lib.context_managers import lockfile_nonblocking
from zerver.lib.initial_password import initial_password
from zerver.models import Client, Realm, UserProfile
from zerver.models.clients import get_client

def is_integer_string(val: str) -> bool:
    try:
        int(val)
        return True
    except ValueError:
        return False

def check_config() -> None:
    for setting_name, default in settings.REQUIRED_SETTINGS:
        try:
            if getattr(settings, setting_name) != default:
                continue
        except AttributeError:
            pass

        raise CommandError(f"Error: You must set {setting_name} in /etc/zulip/settings.py.")

class HandleMethod(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

F = TypeVar("F", bound=Callable[..., None])

def abort_unless_locked(handle_func: F) -> F:
    @wraps(handle_func)
    def our_handle(self: BaseCommand, *args: Any, **kwargs: Any) -> None:
        os.makedirs(settings.LOCKFILE_DIRECTORY, exist_ok=True)
        lockfile_name: str = handle_func.__module__.split(".")[-1]
        lockfile_path: str = settings.LOCKFILE_DIRECTORY + "/" + lockfile_name + ".lock"
        with lockfile_nonblocking(lockfile_path) as lock_acquired:
            if not lock_acquired:  # nocoverage
                self.stdout.write(
                    self.style.ERROR(f"Lock {lockfile_path} is unavailable; exiting.")
                )
                sys.exit(1)
            handle_func(self, *args, **kwargs)
    return our_handle  # type: ignore

@dataclass
class CreateUserParameters:
    email: str
    full_name: str
    password: Optional[str]

class ZulipBaseCommand(BaseCommand):
    @override
    def create_parser(self, prog_name: str, subcommand: str, **kwargs: Any) -> CommandParser:
        parser: CommandParser = super().create_parser(prog_name, subcommand, **kwargs)
        parser.add_argument(
            "--automated",
            help="This command is run non-interactively (enables Sentry, etc)",
            action=BooleanOptionalAction,
            default=not sys.stdin.isatty(),
        )
        parser.formatter_class = RawTextHelpFormatter
        return parser

    @override
    def execute(self, *args: Any, **options: Any) -> None:
        if settings.SENTRY_DSN and not options["automated"]:  # nocoverage
            import sentry_sdk
            sentry_sdk.init()
        super().execute(*args, **options)

    def add_realm_args(
        self, parser: ArgumentParser, *, required: bool = False, help: Optional[str] = None
    ) -> None:
        if help is None:
            help = (
                "The numeric or string ID (subdomain) of the Zulip organization to modify.\n"
                "You can use the command list_realms to find ID of the realms in this server."
            )
        parser.add_argument("-r", "--realm", dest="realm_id", required=required, help=help)

    def add_create_user_args(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "email",
            metavar="<email>",
            nargs="?",
            help="Email address for the new user",
        )
        parser.add_argument(
            "full_name",
            metavar="<full name>",
            nargs="?",
            help="Full name for the new user",
        )
        parser.add_argument(
            "--password",
            help="""\
Password for the new user. Recommended only in a development environment.

Sending passwords via command-line arguments is insecure,
since it can be snooped by any process running on the
server via `ps -ef` or reading bash history. Prefer
--password-file.""",
        )
        parser.add_argument("--password-file", help="File containing a password for the new user.")

    def add_user_list_args(
        self,
        parser: _ActionsContainer,
        help: str = "A comma-separated list of email addresses.",
        all_users_help: str = "All users in realm.",
    ) -> None:
        parser.add_argument("-u", "--users", help=help)
        parser.add_argument("-a", "--all-users", action="store_true", help=all_users_help)

    def get_realm(self, options: dict[str, Any]) -> Optional[Realm]:
        val: Optional[str] = options.get("realm_id")
        if val is None:
            return None

        try:
            if is_integer_string(val):
                return Realm.objects.get(id=val)
            return Realm.objects.get(string_id=val)
        except Realm.DoesNotExist:
            raise CommandError(
                "There is no realm with id '{}'. Aborting.".format(options["realm_id"])
            )

    def get_users(
        self,
        options: dict[str, Any],
        realm: Optional[Realm],
        is_bot: Optional[bool] = None,
        include_deactivated: bool = False,
    ) -> QuerySet[UserProfile]:
        if "all_users" in options:
            all_users: bool = bool(options["all_users"])
            if not options.get("users") and not all_users:
                raise CommandError("You have to pass either -u/--users or -a/--all-users.")
            if options.get("users") and all_users:
                raise CommandError("You can't use both -u/--users and -a/--all-users.")
            if all_users and realm is None:
                raise CommandError("The --all-users option requires a realm; please pass --realm.")
            if all_users:
                user_profiles = UserProfile.objects.filter(realm=realm)
                if not include_deactivated:
                    user_profiles = user_profiles.filter(is_active=True)
                if is_bot is not None:
                    return user_profiles.filter(is_bot=is_bot)
                return user_profiles

        if options.get("users") is None:
            return UserProfile.objects.none()
        emails = {email.strip() for email in options["users"].split(",")}
        for email in emails:
            self.get_user(email, realm)
        user_profiles = UserProfile.objects.all().select_related("realm")
        if realm is not None:
            user_profiles = user_profiles.filter(realm=realm)
        email_matches = [Q(delivery_email__iexact=e) for e in emails]
        user_profiles = user_profiles.filter(reduce(lambda a, b: a | b, email_matches)).order_by("id")
        return user_profiles

    def get_user(self, email: str, realm: Optional[Realm]) -> UserProfile:
        if realm is not None:
            try:
                return UserProfile.objects.select_related("realm").get(
                    delivery_email__iexact=email.strip(), realm=realm
                )
            except UserProfile.DoesNotExist:
                raise CommandError(
                    f"The realm '{realm}' does not contain a user with email '{email}'"
                )
        try:
            return UserProfile.objects.select_related("realm").get(delivery_email__iexact=email.strip())
        except MultipleObjectsReturned:
            raise CommandError(
                "This Zulip server contains multiple users with that email (in different realms);"
                " please pass `--realm` to specify which one to modify."
            )
        except UserProfile.DoesNotExist:
            raise CommandError(f"This Zulip server does not contain a user with email '{email}'")

    def get_client(self) -> Client:
        return get_client("ZulipServer")

    def get_create_user_params(self, options: dict[str, Any]) -> CreateUserParameters:  # nocoverage
        if options.get("email") is None:
            email: str = input("Email: ")
        else:
            email = options["email"]

        try:
            validators.validate_email(email)
        except ValidationError:
            raise CommandError("Invalid email address.")

        if options.get("full_name") is None:
            full_name: str = input("Full name: ")
        else:
            full_name = options["full_name"]

        if options.get("password_file") is not None:
            with open(options["password_file"]) as f:
                password: Optional[str] = f.read().strip()
        elif options.get("password") is not None:
            logging.warning(
                "Passing password on the command line is insecure; prefer --password-file."
            )
            password = options["password"]
        else:
            user_initial_password: Optional[str] = initial_password(email)
            if user_initial_password is None:
                logging.info("User will be created with a disabled password.")
            else:
                assert settings.DEVELOPMENT
                logging.info("Password will be available via `./manage.py print_initial_password`.")
            password = user_initial_password

        return CreateUserParameters(
            email=email,
            full_name=full_name,
            password=password,
        )