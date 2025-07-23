"""
Turn docstrings from a single module into a markdown file.

We do this with PydocMarkdown, using custom processors and renderers defined here.
"""
import argparse
import logging
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union, Pattern, Match, Set, Callable, IO
import docspec
from docspec import ApiObject, Class, Data, Function, Indirection, Module
from docspec_python import format_arglist
from pydoc_markdown import Processor, PydocMarkdown, Resolver
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger('py2md')
BASE_MODULE: str = os.environ.get('BASE_MODULE', 'allennlp')
BASE_SOURCE_LINK: str = os.environ.get('BASE_SOURCE_LINK', 'https://github.com/allenai/allennlp/blob/main/allennlp/')

class DocstringError(Exception):
    pass

def emphasize(s: str) -> str:
    s = s.replace('_', '\\_')
    return f'__{s}__'

class Section(Enum):
    ARGUMENTS = 'ARGUMENTS'
    PARAMETERS = 'PARAMETERS'
    ATTRIBUTES = 'ATTRIBUTES'
    MEMBERS = 'MEMBERS'
    RETURNS = 'RETURNS'
    RAISES = 'RAISES'
    EXAMPLES = 'EXAMPLES'
    OTHER = 'OTHER'

    @classmethod
    def from_str(cls, section: str) -> 'Section':
        section = section.upper()
        for member in cls:
            if section == member.value:
                return member
        return cls.OTHER

REQUIRED_PARAM_RE: Pattern[str] = re.compile('^`([^`]+)`(, required\\.?)?$')
OPTIONAL_PARAM_RE: Pattern[str] = re.compile('^`([^`]+)`,?\\s+(optional,?\\s)?\\(\\s?(optional,\\s)?default\\s?=\\s?`([^`]+)`\\s?\\)\\.?$')
OPTIONAL_PARAM_NO_DEFAULT_RE: Pattern[str] = re.compile('^`([^`]+)`,?\\s+optional\\.?$')

@dataclass
class Param:
    ident: str
    ty: Optional[str] = None
    required: bool = False
    default: Optional[str] = None

    @classmethod
    def from_line(cls, line: str) -> Optional['Param']:
        if ':' not in line:
            return None
        ident, description = line.split(':', 1)
        ident = ident.strip()
        description = description.strip()
        if ' ' in ident:
            return None
        maybe_match = REQUIRED_PARAM_RE.match(description)
        if maybe_match:
            ty = maybe_match.group(1)
            return cls(ident=ident, ty=ty, required=True)
        maybe_match = OPTIONAL_PARAM_RE.match(description)
        if maybe_match:
            ty = maybe_match.group(1)
            default = maybe_match.group(4)
            return cls(ident=ident, ty=ty, required=False, default=default)
        maybe_match = OPTIONAL_PARAM_NO_DEFAULT_RE.match(description)
        if maybe_match:
            ty = maybe_match.group(1)
            return cls(ident=ident, ty=ty, required=False)
        raise DocstringError(f"Invalid parameter / attribute description: '{line}'\nMake sure types are enclosed in backticks.\nRequired parameters should be documented like: '{{ident}} : `{{type}}`'\nOptional parameters should be documented like: '{{ident}} : `{{type}}`, optional (default = `{{expr}}`)'\n")

    def to_line(self) -> str:
        line = f'- {emphasize(self.ident)} :'
        if self.ty:
            line += f' `{self.ty}`'
            if not self.required:
                line += ', optional'
                if self.default:
                    line += f' (default = `{self.default}`)'
        line += ' <br>'
        return line

Attrib = Param

@dataclass
class RetVal:
    description: Optional[str] = None
    ident: Optional[str] = None
    ty: Optional[str] = None

    @classmethod
    def from_line(cls, line: str) -> 'RetVal':
        if ': ' not in line:
            return cls(description=line)
        ident, ty = line.split(':', 1)
        ident = ident.strip()
        ty = ty.strip()
        if ty and (not ty.startswith('`')):
            raise DocstringError(f"Type should be enclosed in backticks: '{line}'")
        return cls(ident=ident, ty=ty)

    def to_line(self) -> str:
        if self.description:
            line = f'- {self.description} <br>'
        elif self.ident:
            line = f'- {emphasize(self.ident)}'
            if self.ty:
                line += f' : {self.ty} <br>'
            else:
                line += ' <br>'
        else:
            raise DocstringError('RetVal must have either description or ident')
        return line

@dataclass
class ProcessorState:
    current_section: Optional[Section] = None
    codeblock_opened: bool = False
    consecutive_blank_line_count: int = 0

@dataclass
class AllenNlpDocstringProcessor(Processor):
    """
    Use to turn our docstrings into Markdown.
    """
    CROSS_REF_RE: Pattern[str] = re.compile('(:(class|func|mod):`~?([a-zA-Z0-9_.]+)`)')
    UNDERSCORE_HEADER_RE: Pattern[str] = re.compile('(.*)\\n-{3,}\\n')
    MULTI_LINE_LINK_RE: Pattern[str] = re.compile('(\\[[^\\]]+\\])\\n\\s*(\\([^\\)]+\\))')

    def process(self, modules: List[Module], resolver: Resolver) -> None:
        docspec.visit(modules, self.process_node)

    def process_node(self, node: ApiObject) -> None:
        if not getattr(node, 'docstring', None):
            return
        lines: List[str] = []
        state: ProcessorState = ProcessorState()
        docstring: str = node.docstring
        docstring = self.UNDERSCORE_HEADER_RE.sub('# \\g<1>', docstring)
        docstring = self.MULTI_LINE_LINK_RE.sub('\\g<1>\\g<2>', docstring)
        for line in docstring.split('\n'):
            if line.startswith('