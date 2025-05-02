"""Generate CLI documentation."""
from __future__ import annotations
import inspect
import logging
import warnings
from pathlib import Path
from typing import TypedDict
import click
import typer
from click import Command, MultiCommand, Parameter
from griffe import Docstring, DocstringSection, DocstringSectionExamples
from jinja2 import Environment, FileSystemLoader, select_autoescape
logging.getLogger('griffe.docstrings.google').setLevel(logging.ERROR)

class ArgumentDict(TypedDict):
    """A dictionary representing a command argument."""

class CommandSummaryDict(TypedDict):
    """A dictionary representing a command summary."""

class BuildDocsContext(TypedDict):
    """A dictionary representing a command context."""

def get_help_text(docstring_object):
    """Get help text sections from a docstring.

    Args:
        docstring_object: The docstring to parse.

    Returns:
        list of docstring text sections.

    """
    return [section for section in Docstring(inspect.cleandoc(docstring_object), lineno=1).parse('google', warnings=False) if section.kind == 'text']

def get_examples(docstring_object):
    """Get example strings from a docstring.

    Args:
        docstring_object: The docstring to parse.

    Returns:
        list of example strings.

    """
    return [text for section in Docstring(inspect.cleandoc(docstring_object), lineno=1).parse('google', warnings=False) if isinstance(section, DocstringSectionExamples) for _, text in section.value]

def build_docs_context(*, obj, ctx, indent=0, name='', call_prefix=''):
    """Build a command context for documentation generation.

    Args:
        obj: The Click command object to document
        ctx: The Click context
        indent: Indentation level for nested commands
        name: Override name for the command
        call_prefix: Prefix to add to command name

    Returns:
        A BuildDocsContext object

    """
    if call_prefix:
        command_name = f'{call_prefix} {obj.name or ''}'.strip()
    else:
        command_name = name if name else obj.name or ''
    title = f'`{command_name}`' if command_name else 'CLI'
    usage_pieces = obj.collect_usage_pieces(ctx)
    args_list = []
    opts_list = []
    for param in obj.get_params(ctx):
        if isinstance(param, click.Option) and '--help' in param.opts:
            continue
        help_record = param.get_help_record(ctx)
        if help_record is not None:
            param_name, param_help = help_record
            if getattr(param, 'param_type_name', '') == 'argument':
                args_list.append({'name': param_name, 'help': param_help})
            elif getattr(param, 'param_type_name', '') == 'option':
                opts_list.append(param)
    commands_list = []
    subcommands = []
    if isinstance(obj, MultiCommand):
        all_commands = obj.list_commands(ctx)
        blocked_commands = {'help', '--help', 'deploy', 'cloud'}
        filtered_commands = [cmd for cmd in all_commands if cmd not in blocked_commands]
        for command in filtered_commands:
            command_obj = obj.get_command(ctx, command)
            assert command_obj, f'Command {command} not found in {obj.name}'
            cmd_name = command_obj.name or ''
            cmd_help = command_obj.get_short_help_str()
            commands_list.append({'name': cmd_name, 'help': cmd_help})
        for command in filtered_commands:
            command_obj = obj.get_command(ctx, command)
            assert command_obj
            sub_ctx = build_docs_context(obj=command_obj, ctx=ctx, indent=indent + 1, name='', call_prefix=command_name)
            subcommands.append(sub_ctx)
    return BuildDocsContext(indent=indent, command_name=command_name, title=title, help=get_help_text(obj.help or ''), examples=get_examples(obj.help or ''), usage_pieces=usage_pieces, args=args_list, opts=opts_list, epilog=obj.epilog, commands=commands_list, subcommands=subcommands)

def escape_mdx(text):
    """Escape characters that commonly break MDX (Mintlify).

    - Replace angle brackets < >
    - Replace curly braces { }
    - Escape backticks, pipes, and arrow functions
    - Escape dollar signs to avoid template interpolation.
    """
    import re
    if not text:
        return ''
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('{', '&#123;').replace('}', '&#125;')
    text = text.replace('`', '\\`')
    text = text.replace('|', '\\|')
    text = re.sub('(?<!\\w)=>(?!\\w)', '\\=>', text)
    text = text.replace('$', '\\$')
    return re.sub('(?m)^!', '\\!', text)

def write_command_docs(command_context, env, output_dir):
    """Render a single command (and do *not* recurse in the template).

    Then recurse here in Python for each subcommand.

    Args:
        command_context: Context containing command documentation
        env: Jinja environment for rendering templates
        output_dir: Directory to write output files

    """
    template = env.get_template('docs_template.jinja')
    rendered = template.render(command=command_context)
    command_name_clean = command_context['command_name'].replace(' ', '_')
    if not command_name_clean:
        command_name_clean = 'cli_root'
    filename = f'{command_name_clean}.mdx'
    filepath = Path(output_dir) / filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with Path.open(filepath, mode='w', encoding='utf-8') as f:
        f.write(rendered)
    for sub_ctx in command_context['subcommands']:
        write_command_docs(sub_ctx, env, output_dir)

def render_command_and_subcommands(cmd_context, env):
    """Render the given command then recurse in Python to render/append all subcommands.

    Args:
        cmd_context: Context containing command documentation
        env: Jinja environment for rendering templates

    Returns:
        Rendered documentation string

    """
    template = env.get_template('docs_template.jinja')
    rendered = template.render(command=cmd_context)
    for sub_ctx in cmd_context['subcommands']:
        sub_rendered = render_command_and_subcommands(sub_ctx, env)
        rendered += '\n\n' + sub_rendered
    return rendered

def write_subcommand_docs(top_level_sub, env, output_dir):
    """Render one *top-level* and all nested subcommands into a single MDX file.

    Args:
        top_level_sub: Context containing top-level command documentation
        env: Jinja environment for rendering templates
        output_dir: Directory to write output files

    """
    content = render_command_and_subcommands(top_level_sub, env)
    name_parts = top_level_sub['command_name'].split()
    file_stub = name_parts[-1] if name_parts else 'cli-root'
    filename = f'{file_stub}.mdx'
    file_path = Path(output_dir) / filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with Path.open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_docs_for_click(*, obj, ctx, indent=0, name='', call_prefix=''):
    """Build the top-level docs context & generate one MDX file per subcommand.

    Args:
        obj: The Click command object to document
        ctx: The Click context
        indent: Indentation level for nested commands
        name: Override name for the command
        call_prefix: Prefix to add to command name
        title: Override title for the command

    Returns:
        Empty string (files are written to disk)

    """
    docs_context = build_docs_context(obj=obj, ctx=ctx, indent=indent, name=name, call_prefix=call_prefix)
    env = Environment(loader=FileSystemLoader('./scripts/templates'), autoescape=select_autoescape(['html', 'xml']))
    env.filters['escape_mdx'] = escape_mdx
    cli_dir = './docs/v3/api-ref/cli'
    for sub_ctx in docs_context['subcommands']:
        write_subcommand_docs(sub_ctx, env, cli_dir)
    return ''
if __name__ == '__main__':
    with warnings.catch_warnings():
        from prefect.cli.root import app
        click_obj = typer.main.get_command(app)
        main_ctx = click.Context(click_obj)
        get_docs_for_click(obj=click_obj, ctx=main_ctx)