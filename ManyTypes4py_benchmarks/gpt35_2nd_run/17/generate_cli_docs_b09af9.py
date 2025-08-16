from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import TypedDict
import click
import typer
from click import Command, MultiCommand, Parameter
from griffe import Docstring, DocstringSection, DocstringSectionExamples

class ArgumentDict(TypedDict):
    """A dictionary representing a command argument."""

class CommandSummaryDict(TypedDict):
    """A dictionary representing a command summary."""

class BuildDocsContext(TypedDict):
    """A dictionary representing a command context."""

def get_help_text(docstring_object: str) -> list[DocstringSection]:
    """Get help text sections from a docstring."""
    
def get_examples(docstring_object: str) -> list[str]:
    """Get example strings from a docstring."""
    
def build_docs_context(*, obj: Command, ctx: click.Context, indent: int = 0, name: str = '', call_prefix: str = '') -> BuildDocsContext:
    """Build a command context for documentation generation."""
    
def escape_mdx(text: str) -> str:
    """Escape characters that commonly break MDX (Mintlify)."""
    
def write_command_docs(command_context: BuildDocsContext, env, output_dir: str):
    """Render a single command (and do *not* recurse in the template)."""
    
def render_command_and_subcommands(cmd_context: BuildDocsContext, env) -> str:
    """Render the given command then recurse in Python to render/append all subcommands."""
    
def write_subcommand_docs(top_level_sub: BuildDocsContext, env, output_dir: str):
    """Render one *top-level* and all nested subcommands into a single MDX file."""
    
def get_docs_for_click(*, obj: Command, ctx: click.Context, indent: int = 0, name: str = '', call_prefix: str = '') -> str:
    """Build the top-level docs context & generate one MDX file per subcommand."""
