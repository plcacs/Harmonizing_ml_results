import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, List
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
    app = Flask(__name__)
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
@click.option('--username', '-u', type=str, help='Admin user')
@click.option('--key', '-K', 'want_key', type=str, help='API key (default=random string)')
@click.option('--scope', 'scopes', multiple=True, type=str, help='List of permissions eg. admin:keys, write:alerts')
@click.option('--duration', metavar='SECONDS', type=int, help='Duration API key is valid')
@click.option('--text', type=str, help='Description of API key use')
@click.option('--customer', type=str, help='Customer')
@click.option('--all', 'all_', is_flag=True, help='Create API keys for all admins')
@click.option('--force', is_flag=True, help='Do not skip if API key already exists')
@with_appcontext
def key(
    username: Optional[str],
    want_key: Optional[str],
    scopes: Tuple[str, ...],
    duration: Optional[int],
    text: Optional[str],
    customer: Optional[str],
    all_: bool,
    force: bool
) -> None:
    """
    Create an admin API key.
    """
    if username and username not in current_app.config['ADMIN_USERS']:
        raise click.UsageError(f'User {username} not an admin')
    if all_ and want_key:
        raise click.UsageError('Can only set API key with "--username".')
    scopes_enum = [Scope(s) for s in scopes] or [Scope.admin, Scope.write, Scope.read]
    expires = datetime.utcnow() + timedelta(seconds=duration) if duration else None
    description = text or 'Created by alertad script'

    def create_key(admin: str, key: Optional[str] = None) -> Optional[ApiKey]:
        api_key = ApiKey(
            user=admin,
            key=key,
            scopes=scopes_enum,
            expire_time=expires,
            text=description,
            customer=customer
        )
        try:
            api_key = api_key.create()
        except Exception as e:
            click.echo(f'ERROR: {e}')
            return None
        else:
            return api_key

    if all_:
        for admin in current_app.config['ADMIN_USERS']:
            keys = [k for k in ApiKey.find_by_user(admin) if k.scopes == scopes_enum]
            if keys and not force:
                api_key = keys[0]
            else:
                api_key = create_key(admin)
            if api_key:
                click.echo(f'{api_key.key:40} {api_key.user}')
    elif username:
        keys = [k for k in ApiKey.find_by_user(username) if k.scopes == scopes_enum]
        if want_key:
            found_keys = [k for k in keys if k.key == want_key]
            if found_keys:
                api_key = found_keys[0]
            else:
                api_key = create_key(username, key=want_key)
        elif keys and not force:
            api_key = keys[0]
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
            admin_keys = [k for k in ApiKey.find_by_user(admin) if Scope.admin in k.scopes]
        except Exception as e:
            click.echo(f'ERROR: {e}')
        else:
            for key in admin_keys:
                click.echo(f'{key.key:40} {key.user}')


class CommandWithOptionalPassword(click.Command):

    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        for i, arg in enumerate(args):
            if arg == '--password':
                try:
                    password = args[i + 1] if not args[i + 1].startswith('--') else None
                except IndexError:
                    password = None
                if not password:
                    password = click.prompt('Password', hide_input=True, confirmation_prompt=True)
                    args.insert(i + 1, password)
        return super().parse_args(ctx, args)


@cli.command('user', cls=CommandWithOptionalPassword, short_help='Create admin user')
@click.option('--name', type=str, help='Name of admin (default=email)')
@click.option('--email', '--username', type=str, help='Email address (login username)')
@click.option('--password', type=str, help='Password (will prompt if not supplied)')
@click.option('--text', type=str, help='Description of admin')
@click.option('--all', 'all_', is_flag=True, help='Create users for all admins')
@with_appcontext
def user(
    name: Optional[str],
    email: Optional[str],
    password: Optional[str],
    text: Optional[str],
    all_: bool
) -> None:
    """
    Create admin users (BasicAuth only).
    """
    if current_app.config['AUTH_PROVIDER'] != 'basic':
        raise click.UsageError(f'Not required for {current_app.config["AUTH_PROVIDER"]} admin users')
    if email and email not in current_app.config['ADMIN_USERS']:
        raise click.UsageError(f'User {email} not an admin')
    if (email or all_) and not password:
        password = click.prompt('Password', hide_input=True)
    description = text or 'Created by alertad script'

    def create_user_obj(name: Optional[str], login: str) -> Optional[User]:
        email_field = login if '@' in login else None
        user = User(
            name=name or login,
            login=login,
            password=generate_password_hash(password),
            roles=current_app.config['ADMIN_ROLES'],
            text=description,
            email=email_field,
            email_verified=bool(email_field)
        )
        try:
            user = user.create()
        except Exception as e:
            click.echo(f'ERROR: {e}')
            return None
        else:
            return user

    if all_:
        for admin in current_app.config['ADMIN_USERS']:
            user_obj = User.find_by_username(admin)
            if not user_obj:
                user_obj = create_user_obj(name=name, login=admin)
            if user_obj:
                click.echo(f'{user_obj.id} {user_obj.login}')
    elif email:
        user_obj = create_user_obj(name, login=email)
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
            user_obj = User.find_by_username(admin)
        except Exception as e:
            click.echo(f'ERROR: {e}')
        else:
            if user_obj:
                click.echo(f'{user_obj.id} {user_obj.login}')
