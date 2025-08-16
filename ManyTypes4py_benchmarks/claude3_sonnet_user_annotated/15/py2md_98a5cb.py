#!/usr/bin/env python

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
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union, cast

import docspec
from docspec import ApiObject, Class, Data, Function, Indirection, Module
from docspec_python import format_arglist
from pydoc_markdown import Processor, PydocMarkdown, Resolver
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("py2md")
BASE_MODULE: str = os.environ.get("BASE_MODULE", "allennlp")
BASE_SOURCE_LINK: str = os.environ.get(
    "BASE_SOURCE_LINK", "https://github.com/allenai/allennlp/blob/main/allennlp/"
)


class DocstringError(Exception):
    pass


def emphasize(s: str) -> str:
    # Need to escape underscores.
    s = s.replace("_", "\\_")
    return f"__{s}__"


class Section(Enum):
    ARGUMENTS = "ARGUMENTS"
    PARAMETERS = "PARAMETERS"
    ATTRIBUTES = "ATTRIBUTES"
    MEMBERS = "MEMBERS"
    RETURNS = "RETURNS"
    RAISES = "RAISES"
    EXAMPLES = "EXAMPLES"
    OTHER = "OTHER"

    @classmethod
    def from_str(cls, section: str) -> "Section":
        section = section.upper()
        for member in cls:
            if section == member.value:
                return member
        return cls.OTHER


REQUIRED_PARAM_RE = re.compile(r"^`([^`]+)`(, required\.?)?$")

OPTIONAL_PARAM_RE = re.compile(
    r"^`([^`]+)`,?\s+(optional,?\s)?\(\s?(optional,\s)?default\s?=\s?`([^`]+)`\s?\)\.?$"
)

OPTIONAL_PARAM_NO_DEFAULT_RE = re.compile(r"^`([^`]+)`,?\s+optional\.?$")


@dataclass
class Param:
    ident: str
    ty: Optional[str] = None
    required: bool = False
    default: Optional[str] = None

    @classmethod
    def from_line(cls, line: str) -> Optional["Param"]:
        if ":" not in line:
            return None

        ident, description = line.split(":", 1)
        ident = ident.strip()
        description = description.strip()

        if " " in ident:
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

        raise DocstringError(
            f"Invalid parameter / attribute description: '{line}'\n"
            "Make sure types are enclosed in backticks.\n"
            "Required parameters should be documented like: '{ident} : `{type}`'\n"
            "Optional parameters should be documented like: '{ident} : `{type}`, optional (default = `{expr}`)'\n"
        )

    def to_line(self) -> str:
        line: str = f"- {emphasize(self.ident)} :"
        if self.ty:
            line += f" `{self.ty}`"
            if not self.required:
                line += ", optional"
                if self.default:
                    line += f" (default = `{self.default}`)"
        line += " <br>"
        return line


# For now we handle attributes / members in the same way as parameters / arguments.
Attrib = Param


@dataclass
class RetVal:
    description: Optional[str] = None
    ident: Optional[str] = None
    ty: Optional[str] = None

    @classmethod
    def from_line(cls, line: str) -> "RetVal":
        if ": " not in line:
            return cls(description=line)
        ident, ty = line.split(":", 1)
        ident = ident.strip()
        ty = ty.strip()
        if ty and not ty.startswith("`"):
            raise DocstringError(f"Type should be enclosed in backticks: '{line}'")
        return cls(ident=ident, ty=ty)

    def to_line(self) -> str:
        if self.description:
            line = f"- {self.description} <br>"
        elif self.ident:
            line = f"- {emphasize(self.ident)}"
            if self.ty:
                line += f" : {self.ty} <br>"
            else:
                line += " <br>"
        else:
            raise DocstringError("RetVal must have either description or ident")
        return line


@dataclass
class ProcessorState:
    parameters: "OrderedDict[str, Param]"
    current_section: Optional[Section] = None
    codeblock_opened: bool = False
    consecutive_blank_line_count: int = 0


@dataclass
class AllenNlpDocstringProcessor(Processor):
    """
    Use to turn our docstrings into Markdown.
    """

    CROSS_REF_RE = re.compile("(:(class|func|mod):`~?([a-zA-Z0-9_.]+)`)")
    UNDERSCORE_HEADER_RE = re.compile(r"(.*)\n-{3,}\n")
    MULTI_LINE_LINK_RE = re.compile(r"(\[[^\]]+\])\n\s*(\([^\)]+\))")

    def process(self, modules: List[Module], resolver: Optional[Resolver]) -> None:
        docspec.visit(modules, self.process_node)

    def process_node(self, node: docspec.ApiObject) -> None:
        if not getattr(node, "docstring", None):
            return

        lines: List[str] = []
        state: ProcessorState = ProcessorState(parameters=OrderedDict())

        docstring = node.docstring

        # Standardize header syntax to use '#' instead of underscores.
        docstring = self.UNDERSCORE_HEADER_RE.sub(r"# \g<1>", docstring)

        # It's common to break up markdown links into multiple lines in docstrings, but
        # they won't render as links in the doc HTML unless they are all on one line.
        docstring = self.MULTI_LINE_LINK_RE.sub(r"\g<1>\g<2>", docstring)

        for line in docstring.split("\n"):
            # Check if we're starting or ending a codeblock.
            if line.startswith("