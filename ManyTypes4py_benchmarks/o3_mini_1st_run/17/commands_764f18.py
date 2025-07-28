#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple
import click
from flask import Flask, current_app
from flask.cli import FlaskGroup, ScriptInfo, with_appcontext
from alerta.app import config, create_app, db, key_helper, qb
from alerta.auth.utils import generate_password_hash
from alerta.models.enums import Scope
from alerta.models.key import ApiKey
from alerta.models.user import User
from alerta.version import __version__


def _create_app(config_override: Optional[Dict[str, Any]] = None, environment: Optional[str] = None) -> Flask:
    app: Flask = Flask(__name__)
    app.config['ENVIRONMENT'] = environment
    config.init_app(app)
    app.config.update(config_override or {})
    db.init_db(app)
    qb.init_app(app)
    key_helper.init_app(app)
    return app


@click.group(cls=FlaskGroup, add_version_option=False)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    Management command-line tool for Alerta server.
    """
    if ctx.invoked_subcommand in ['routes', 'run', 'shell']:
        ctx.obj = ScriptInfo(create_app=create_app)
    else:
        ctx.obj = ScriptInfo(create_app=_create_app)


@cli.command('key', short_help='Create an admin API key')
@click.option('--username', '-u', help='Admin user')
@click.option('--key', '-K', 'want_key', help='API key (default=random string)')
@click.option('--scope', 'scopes', multiple=True, help='List of permissions eg. admin:keys, write:alerts')
@click.option('--duration', metavar='SECONDS', type=int, help='Duration API key is valid')
@click.option('--text', help='Description of API key use')
@click.option('--customer', help='Customer')
@click.option('--all', is_flag=True, help='Create API keys for all admins')
@click.option('--force', is_flag=True, help='Do not skip if API key already exists')
@with_appcontext
def key(username: Optional[str],
        want_key: Optional[str],
        scopes: Tuple[str, ...],
        duration: Optional[int],
        text: Optional[str],
        customer: Optional[str],
        all: bool,
        force: bool) -> None:
    """
    Create an admin API key.
    """
    if username and username not in current_app.config['ADMIN_USERS']:
        raise click.UsageError(f'User {username} not an admin')
    if all and want_key:
        raise click.UsageError('Can only set API key with "--username".')
    scopes_list: List[Scope] = [Scope(s) for s in scopes] or [Scope.admin, Scope.write, Scope.read]
    expires: Optional[datetime] = datetime.utcnow() + timedelta(seconds=duration) if duration else None
    text = text or 'Created by alertad script'

    def create_key(admin: str, key: Optional[str] = None) -> Optional[ApiKey]:
        api_key: ApiKey = ApiKey(user=admin, key=key, scopes=scopes_list, expire_time=expires, text=text, customer=customer)
        try:
            api_key = api_key.create()
        except Exception as e:
            click.echo(f'ERROR: {e}')
            return None
        else:
            return api_key

    if all:
        for admin in current_app.config['ADMIN_USERS']:
            keys_found: List[ApiKey] = [k for k in ApiKey.find_by_user(admin) if k.scopes == scopes_list]
            if keys_found and (not force):
                api_key = keys_found[0]
            else:
                api_key = create_key(admin)
            if api_key:
                click.echo(f'{api_key.key:40} {api_key.user}')
    elif username:
        keys_found: List[ApiKey] = [k for k in ApiKey.find_by_user(username) if k.scopes == scopes_list]
        if want_key:
            found_key: List[ApiKey] = [k for k in keys_found if k.key == want_key]
            if found_key:
                api_key = found_key[0]
            else:
                api_key = create_key(username, key=want_key)
        elif keys_found and (not force):
            api_key = keys_found[0]
        else:
            api_key = create_key(username)
        if api_key:
            click.echo(api_key.key)
        else:
            sys.exit(1)
    else:
        raise click.UsageError("Must set '--username' or use '--all'")


@cli.command('keys', short_help='List admin API keys')
@with_appcontext
def keys() -> None:
    """
    List admin API keys.
    """
    for admin in current_app.config['ADMIN_USERS']:
        try:
            keys_list: List[ApiKey] = [k for k in ApiKey.find_by_user(admin) if Scope.admin in k.scopes]
        except Exception as e:
            click.echo(f'ERROR: {e}')
        else:
            for api_key in keys_list:
                click.echo(f'{api_key.key:40} {api_key.user}')


class CommandWithOptionalPassword(click.Command):

    def parse_args(self, ctx: click.Context, args: List[str]) -> None:
        for i, a in enumerate(args):
            if a == '--password':
                try:
                    password = args[i + 1] if not args[i + 1].startswith('--') else None
                except IndexError:
                    password = None
                if not password:
                    password = click.prompt('Password', hide_input=True, confirmation_prompt=True)
                    args.insert(i + 1, password)
        super().parse_args(ctx, args)
        return None


@cli.command('user', cls=CommandWithOptionalPassword, short_help='Create admin user')
@click.option('--name', help='Name of admin (default=email)')
@click.option('--email', '--username', help='Email address (login username)')
@click.option('--password', help='Password (will prompt if not supplied)')
@click.option('--text', help='Description of admin')
@click.option('--all', is_flag=True, help='Create users for all admins')
@with_appcontext
def user(name: Optional[str],
         email: Optional[str],
         password: Optional[str],
         text: Optional[str],
         all: bool) -> None:
    """
    Create admin users (BasicAuth only).
    """
    if current_app.config['AUTH_PROVIDER'] != 'basic':
        raise click.UsageError(f"Not required for {current_app.config['AUTH_PROVIDER']} admin users")
    if email and email not in current_app.config['ADMIN_USERS']:
        raise click.UsageError(f'User {email} not an admin')
    if (email or all) and (not password):
        password = click.prompt('Password', hide_input=True)
    text = text or 'Created by alertad script'

    def create_user(name: Optional[str], login: str) -> Optional[User]:
        email_value: Optional[str] = login if '@' in login else None
        user_obj: User = User(name=name or login,
                              login=login,
                              password=generate_password_hash(password),  # type: ignore
                              roles=current_app.config['ADMIN_ROLES'],
                              text=text,
                              email=email_value,
                              email_verified=bool(email_value))
        try:
            user_obj = user_obj.create()
        except Exception as e:
            click.echo(f'ERROR: {e}')
            return None
        else:
            return user_obj

    if all:
        for admin in current_app.config['ADMIN_USERS']:
            user_obj: Optional[User] = User.find_by_username(admin)
            if not user_obj:
                user_obj = create_user(name=admin, login=admin)
            if user_obj:
                click.echo(f'{user_obj.id} {user_obj.login}')
    elif email:
        user_obj = create_user(name, login=email)
        if user_obj:
            click.echo(user_obj.id)
        else:
            sys.exit(1)
    else:
        raise click.UsageError("Must set '--email' or use '--all'")


@cli.command('users', short_help='List admin users')
@with_appcontext
def users() -> None:
    """
    List admin users.
    """
    for admin in current_app.config['ADMIN_USERS']:
        try:
            user_obj: Optional[User] = User.find_by_username(admin)
        except Exception as e:
            click.echo(f'ERROR: {e}')
        else:
            if user_obj:
                click.echo(f'{user_obj.id} {user_obj.login}')