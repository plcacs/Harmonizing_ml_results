from typing import Any, Protocol
from django.core.management.base import BaseCommand, CommandError
from django.db.models import QuerySet
from zerver.lib.context_managers import lockfile_nonblocking
from zerver.models import Realm, UserProfile
from zerver.models.clients import get_client

def is_integer_string(val: str) -> bool:
    try:
        int(val)
        return True
    except ValueError:
        return False

def check_config() -> None:
    ...

class HandleMethod(Protocol):
    def __call__(self, *args, **kwargs) -> Any:
        ...

def abort_unless_locked(handle_func: HandleMethod) -> HandleMethod:
    ...

class CreateUserParameters:
    email: str
    full_name: str
    password: str

class ZulipBaseCommand(BaseCommand):
    def create_parser(self, prog_name: str, subcommand: str, **kwargs: Any) -> ArgumentParser:
        ...
    
    def execute(self, *args: Any, **options: Any) -> None:
        ...
    
    def add_realm_args(self, parser: CommandParser, *, required: bool = False, help: str = None) -> None:
        ...
    
    def add_create_user_args(self, parser: CommandParser) -> None:
        ...
    
    def add_user_list_args(self, parser: CommandParser, help: str = 'A comma-separated list of email addresses.', all_users_help: str = 'All users in realm.') -> None:
        ...
    
    def get_realm(self, options: dict) -> Realm:
        ...
    
    def get_users(self, options: dict, realm: Realm, is_bot: bool = None, include_deactivated: bool = False) -> QuerySet[UserProfile]:
        ...
    
    def get_user(self, email: str, realm: Realm) -> UserProfile:
        ...
    
    def get_client(self) -> Any:
        ...
    
    def get_create_user_params(self, options: dict) -> CreateUserParameters:
        ...
