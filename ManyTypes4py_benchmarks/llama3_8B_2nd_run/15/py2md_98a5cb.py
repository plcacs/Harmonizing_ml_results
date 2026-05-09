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
from typing import List, Optional, Tuple
import docspec
from docspec import ApiObject, Class, Data, Function, Indirection, Module
from docspec_python import format_arglist
from pydoc_markdown import Processor, PydocMarkdown, Resolver
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer

@dataclass
class Param:
    ty: Optional[str]
    required: bool
    default: Optional[str]

    @classmethod
    def from_line(cls, line: str) -> Optional['Param']:
        # ...

    def to_line(self) -> str:
        # ...

@dataclass
class RetVal:
    description: Optional[str]
    ident: Optional[str]
    ty: Optional[str]

    @classmethod
    def from_line(cls, line: str) -> Optional['RetVal']:
        # ...

    def to_line(self) -> str:
        # ...

@dataclass
class ProcessorState:
    current_section: Optional[Section]
    codeblock_opened: bool
    consecutive_blank_line_count: int

@dataclass
class AllenNlpDocstringProcessor(Processor):
    CROSS_REF_RE: re.Pattern
    UNDERSCORE_HEADER_RE: re.Pattern
    MULTI_LINE_LINK_RE: re.Pattern

    def process(self, modules: List[Module], resolver: Resolver) -> None:
        # ...

    def _preprocess_line(self, node: ApiObject, line: str, state: ProcessorState) -> str:
        # ...

    def _transform_cross_references(self, line: str) -> str:
        # ...

@dataclass
class AllenNlpFilterProcessor(Processor):
    PRIVATE_METHODS_TO_KEEP: frozenset

    def process(self, modules: List[Module], resolver: Resolver) -> None:
        # ...

    def _check(self, node: ApiObject) -> bool:
        # ...

class AllenNlpRenderer(MarkdownRenderer):
    def _format_function_signature(self, func: Function, override_name: Optional[str] = None, add_method_bar: bool = True, include_parent_class: bool = True) -> str:
        # ...

    def _format_data_signature(self, data: Data) -> str:
        # ...

    def _format_classdef_signature(self, cls: Class) -> str:
        # ...

    def _render_module_breadcrumbs(self, fp: IO, mod: Module) -> None:
        # ...

    def _render_object(self, fp: IO, level: int, obj: ApiObject) -> None:
        # ...

def py2md(module: str, out: Optional[str] = None) -> bool:
    # ...

def _py2md_wrapper(x: Tuple[str, Optional[str]]) -> bool:
    # ...

def parse_args() -> argparse.Namespace:
    # ...

def main() -> None:
    # ...
