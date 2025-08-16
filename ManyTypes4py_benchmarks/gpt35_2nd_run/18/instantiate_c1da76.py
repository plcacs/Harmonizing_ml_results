import os
import re
import tempfile
import operator
import contextlib
from pathlib import Path
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import testing
from . import utils

LINETOKEN: str = '@nevergrad' + '@'
COMMENT_CHARS: dict = {'.c': '//', '.h': '//', '.cpp': '//', '.hpp': '//', '.py': '#', '.m': '%'}

def _convert_to_string(data: tp.Any, extension: str) -> str:
    ...

class Placeholder:
    pattern: str = 'NG_ARG' + '{(?P<name>\\w+?)(\\|(?P<comment>.+?))?}'

    def __init__(self, name: str, comment: str):
        ...

    @classmethod
    def finditer(cls, text: str) -> list:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: 'Placeholder') -> bool:
        ...

    @classmethod
    def sub(cls, text: str, extension: str, replacers: dict) -> str:
        ...

def symlink_folder_tree(folder: str, shadow_folder: str) -> None:
    ...

def uncomment_line(line: str, extension: str) -> str:
    ...

class FileTextFunction:
    def __init__(self, filepath: Path):
        ...

    def __call__(self, **kwargs: tp.Any) -> str:
        ...

    def __repr__(self) -> str:
        ...

class FolderInstantiator:
    def __init__(self, folder: str, clean_copy: bool = False):
        ...

    def instantiate_to_folder(self, outfolder: str, kwargs: dict) -> None:
        ...

    @contextlib.contextmanager
    def instantiate(self, **kwargs: tp.Any) -> tp.Any:
        ...

class FolderFunction:
    def __init__(self, folder: str, command: list, verbose: bool = False, clean_copy: bool = False):
        ...

    @staticmethod
    def register_file_type(suffix: str, comment_chars: str) -> None:
        ...

    def __call__(self, **kwargs: tp.Any) -> tp.Any:
        ...

def get_last_line_as_float(output: str) -> float:
    ...
