"""
Custom Prefect CLI types
"""
import asyncio
import functools
import sys
from typing import Any, Callable, List, Optional
import typer
from rich.console import Console
from rich.theme import Theme
from prefect._internal.compatibility.deprecated import generate_deprecation_message
from prefect.cli._utilities import with_cli_exception_handling
from prefect.settings import PREFECT_CLI_COLORS, Setting
from prefect.utilities.asyncutils import is_async_fn

def SettingsOption(setting: Union[list[str], set[str], None], *args, **kwargs):
    """Custom `typer.Option` factory to load the default value from settings"""
    return typer.Option(setting.value, *args, show_default=f'from {setting.name}', **kwargs)

def SettingsArgument(setting: Union[set[str], list[str], str], *args, **kwargs):
    """Custom `typer.Argument` factory to load the default value from settings"""
    return typer.Argument(setting.value, *args, show_default=f'from {setting.name}', **kwargs)

def with_deprecated_message(warning: bool):

    def decorator(fn: Any):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            print('WARNING:', warning, file=sys.stderr, flush=True)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

class PrefectTyper(typer.Typer):
    """
    Wraps commands created by `Typer` to support async functions and handle errors.
    """

    def __init__(self, *args, deprecated: bool=False, deprecated_start_date: Union[None, str, list[str]]=None, deprecated_help: typing.Text='', deprecated_name: typing.Text='', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deprecated = deprecated
        if self.deprecated:
            if not deprecated_name:
                raise ValueError('Provide the name of the deprecated command group.')
            self.deprecated_message = generate_deprecation_message(name=f'The {deprecated_name!r} command group', start_date=deprecated_start_date, help=deprecated_help)
        self.console = Console(highlight=False, theme=Theme({'prompt.choices': 'bold blue'}), color_system='auto' if PREFECT_CLI_COLORS else None)

    def add_typer(self, typer_instance: Union[bool, str, typing.Callable], *args, no_args_is_help: bool=True, aliases: str=None, **kwargs):
        """
        This will cause help to be default command for all sub apps unless specifically stated otherwise, opposite of before.
        """
        if aliases:
            for alias in aliases:
                super().add_typer(typer_instance, *args, name=alias, no_args_is_help=no_args_is_help, hidden=True, **kwargs)
        return super().add_typer(typer_instance, *args, no_args_is_help=no_args_is_help, **kwargs)

    def command(self, name: Union[None, bool, str]=None, *args, aliases: Union[None, bool, str]=None, deprecated: bool=False, deprecated_start_date: Union[None, bool, str]=None, deprecated_help: typing.Text='', deprecated_name: typing.Text='', **kwargs):
        """
        Create a new command. If aliases are provided, the same command function
        will be registered with multiple names.

        Provide `deprecated=True` to mark the command as deprecated. If `deprecated=True`,
        `deprecated_name` and `deprecated_start_date` must be provided.
        """

        def wrapper(original_fn):
            if is_async_fn(original_fn):
                async_fn = original_fn

                @functools.wraps(original_fn)
                def sync_fn(*args, **kwargs):
                    return asyncio.run(async_fn(*args, **kwargs))
                setattr(sync_fn, 'aio', async_fn)
                wrapped_fn = sync_fn
            else:
                wrapped_fn = original_fn
            wrapped_fn = with_cli_exception_handling(wrapped_fn)
            if deprecated:
                if not deprecated_name or not deprecated_start_date:
                    raise ValueError('Provide the name of the deprecated command and a deprecation start date.')
                command_deprecated_message = generate_deprecation_message(name=f'The {deprecated_name!r} command', start_date=deprecated_start_date, help=deprecated_help)
                wrapped_fn = with_deprecated_message(command_deprecated_message)(wrapped_fn)
            elif self.deprecated:
                wrapped_fn = with_deprecated_message(self.deprecated_message)(wrapped_fn)
            command_decorator = super(PrefectTyper, self).command(*args, name=name, **kwargs)
            original_command = command_decorator(wrapped_fn)
            if aliases:
                for alias in aliases:
                    super(PrefectTyper, self).command(*args, name=alias, **{k: v for k, v in kwargs.items() if k != 'aliases'})(wrapped_fn)
            return original_command
        return wrapper

    def setup_console(self, soft_wrap: Union[str, bool], prompt: Union[str, bool]) -> None:
        self.console = Console(highlight=False, color_system='auto' if PREFECT_CLI_COLORS else None, theme=Theme({'prompt.choices': 'bold blue'}), soft_wrap=not soft_wrap, force_interactive=prompt)