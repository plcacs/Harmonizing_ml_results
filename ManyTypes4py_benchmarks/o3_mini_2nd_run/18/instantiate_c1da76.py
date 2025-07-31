#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import tempfile
import operator
import contextlib
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Set, Union, Generator
import nevergrad.common.typing as tp
from nevergrad.common import testing
from . import utils

LINETOKEN: str = '@nevergrad' + '@'
COMMENT_CHARS: Dict[str, str] = {
    '.c': '//',
    '.h': '//',
    '.cpp': '//',
    '.hpp': '//',
    '.py': '#',
    '.m': '%'
}


def _convert_to_string(data: Any, extension: str) -> str:
    """Converts the data into a string to be injected in a file"""
    if isinstance(data, np.ndarray):
        string: str = repr(data.tolist())
    else:
        string = repr(data)
    if extension in ['.h', '.hpp', '.cpp', '.c'] and isinstance(data, np.ndarray):
        string = string.replace('[', '{').replace(']', '}')
    return string


class Placeholder:
    """Placeholder tokens for external code instrumentation"""
    pattern: str = 'NG_ARG' + '{(?P<name>\\w+?)(\\|(?P<comment>.+?))?}'

    def __init__(self, name: str, comment: Optional[str]) -> None:
        self.name: str = name
        self.comment: Optional[str] = comment

    @classmethod
    def finditer(cls, text: str) -> List[Placeholder]:
        prog = re.compile(cls.pattern)
        return [cls(x.group('name'), x.group('comment')) for x in prog.finditer(text)]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name!r}, {self.comment!r})'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Placeholder) and self.__class__ == other.__class__:
            return (self.name, self.comment) == (other.name, other.comment)
        return False

    @classmethod
    def sub(cls, text: str, extension: str, replacers: Mapping[str, Any]) -> str:
        found: Set[str] = set()
        kwargs: Dict[str, str] = {x: _convert_to_string(y, extension) for x, y in replacers.items()}

        def _replacer(regex: re.Match) -> str:
            name: str = regex.group('name')
            if name in found:
                raise RuntimeError(f'Trying to remplace a second time placeholder "{name}"')
            if name not in kwargs:
                raise KeyError(f'Could not find a value for placeholder "{name}"')
            found.add(name)
            return str(kwargs[name])
        text = re.sub(cls.pattern, _replacer, text)
        missing: Set[str] = set(kwargs) - found
        if missing:
            raise RuntimeError(f'All values have not been consumed: {missing}')
        return text


def symlink_folder_tree(folder: Union[str, Path], shadow_folder: Union[str, Path]) -> None:
    """Utility for copying the tree structure of a folder and symlinking all files."""
    folder_path: Path = Path(folder).expanduser().resolve().absolute()
    shadow_folder_path: Path = Path(shadow_folder).expanduser().resolve().absolute()
    shadow_folder_path.mkdir(parents=True, exist_ok=True)
    for fp in folder_path.iterdir():
        shadow_fp: Path = shadow_folder_path / fp.name
        if fp.is_dir():
            symlink_folder_tree(fp, shadow_fp)
        elif not shadow_fp.exists():
            shadow_fp.symlink_to(fp)


def uncomment_line(line: str, extension: str) -> str:
    if extension not in COMMENT_CHARS:
        raise RuntimeError(f'Unknown file type: {extension}\nDid you register it using {FolderFunction.register_file_type.__name__}?')
    pattern: str = '^(?P<indent> *)'
    pattern += '(?P<linetoken>' + COMMENT_CHARS[extension] + ' *' + LINETOKEN + ' *)'
    pattern += '(?P<command>.*)'
    lineseg: Optional[re.Match] = re.search(pattern, line)
    if lineseg is not None:
        line = lineseg.group('indent') + lineseg.group('command')
    if LINETOKEN in line:
        raise RuntimeError(f'Uncommenting failed for line of {extension} file (a {LINETOKEN} tag remains):\n{line}\nDid you follow the pattern indent+comment+{LINETOKEN}+code (with nothing before the indent)?')
    return line


class FileTextFunction:
    """Function created from a file and generating the text file after
    replacement of the placeholders.
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath: Path = filepath
        assert filepath.exists(), f'{filepath} does not exist'
        with filepath.open('r') as f:
            text: str = f.read()
        deprecated_placeholders: List[str] = ['NG_G{', 'NG_OD{', 'NG_SC{']
        if any(x in text for x in deprecated_placeholders):
            raise RuntimeError(
                f'Found one of deprecated placeholders {deprecated_placeholders}. The API has now evolved to a single placeholder NG_ARG{{name|comment}}, and FolderFunction now takes as many kwargs as placeholders and must be instrumented before optimization.\nPlease refer to the README, PR #73 or issue #45 for more information'
            )
        if LINETOKEN in text:
            lines: List[str] = text.splitlines()
            ext: str = filepath.suffix.lower()
            lines = [l if LINETOKEN not in l else uncomment_line(l, ext) for l in lines]
            text = '\n'.join(lines)
        self.placeholders: List[Placeholder] = Placeholder.finditer(text)
        self._text: str = text
        self.parameters: Set[str] = set()
        for x in self.placeholders:
            if x.name not in self.parameters:
                self.parameters.add(x.name)
            else:
                raise RuntimeError(f'Found duplicate placeholder (names must be unique) with name "{x.name}" in file:\n{self.filepath}')

    def __call__(self, **kwargs: Any) -> str:
        testing.assert_set_equal(set(kwargs.keys()), self.parameters, err_msg='Wrong input parameters.')
        filtered_kwargs: Dict[str, Any] = {x: y for x, y in kwargs.items() if x in self.parameters}
        return Placeholder.sub(self._text, self.filepath.suffix, replacers=filtered_kwargs)

    def __repr__(self) -> str:
        names: List[str] = sorted(self.parameters)
        return f'{self.__class__.__name__}({self.filepath})({", ".join(names)})'


class FolderInstantiator:
    """Folder with instrumentation tokens, which can be instantiated.
    """

    def __init__(self, folder: Union[str, Path], clean_copy: bool = False) -> None:
        self._clean_copy: Optional[utils.TemporaryDirectoryCopy] = None
        self.folder: Path = Path(folder).expanduser().absolute()
        assert self.folder.exists(), f'{folder} does not seem to exist'
        if clean_copy:
            self._clean_copy = utils.TemporaryDirectoryCopy(str(folder))
            self.folder = self._clean_copy.copyname
        self.file_functions: List[FileTextFunction] = []
        names: Set[str] = set()
        for fp in self.folder.glob('**/*'):
            if fp.is_file() and fp.suffix.lower() in COMMENT_CHARS:
                file_func = FileTextFunction(fp)
                fnames: Set[str] = {ph.name for ph in file_func.placeholders}
                if fnames:
                    if fnames & names:
                        raise RuntimeError(f'Found {fp} placeholders in another file (names must be unique): {fnames & names}')
                    self.file_functions.append(file_func)
                    names.update(fnames)
        assert self.file_functions, 'Found no file with placeholders'
        self.file_functions = sorted(self.file_functions, key=operator.attrgetter('filepath'))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.folder}") with files:\n{self.file_functions}'

    @property
    def placeholders(self) -> List[Placeholder]:
        return [p for f in self.file_functions for p in f.placeholders]

    def instantiate_to_folder(self, outfolder: Union[str, Path], kwargs: Mapping[str, Any]) -> None:
        testing.assert_set_equal(set(kwargs.keys()), {x.name for x in self.placeholders}, err_msg='Wrong input parameters.')
        outfolder_path: Path = Path(outfolder).expanduser().absolute()
        assert outfolder_path != self.folder, 'Do not instantiate on same folder!'
        symlink_folder_tree(self.folder, outfolder_path)
        for file_func in self.file_functions:
            inst_fp: Path = outfolder_path / file_func.filepath.relative_to(self.folder)
            os.remove(str(inst_fp))
            with inst_fp.open('w') as f:
                # Filter the kwargs relevant to this file
                file_kwargs: Dict[str, Any] = {x: y for x, y in kwargs.items() if x in file_func.parameters}
                f.write(file_func(**file_kwargs))

    @contextlib.contextmanager
    def instantiate(self, **kwargs: Any) -> Generator[Path, None, None]:
        with tempfile.TemporaryDirectory() as tempfolder:
            subtempfolder: Path = Path(tempfolder) / self.folder.name
            self.instantiate_to_folder(subtempfolder, kwargs)
            yield subtempfolder


class FolderFunction:
    """Turns a folder into a parametrized function (with nevergrad tokens).
    """

    def __init__(self, folder: Union[str, Path], command: List[str], verbose: bool = False, clean_copy: bool = False) -> None:
        self.command: List[str] = command
        self.verbose: bool = verbose
        self.postprocessings: List[tp.Callable[[str], Any]] = [get_last_line_as_float]
        self.instantiator: FolderInstantiator = FolderInstantiator(folder, clean_copy=clean_copy)
        self.last_full_output: Optional[str] = None

    @staticmethod
    def register_file_type(suffix: str, comment_chars: str) -> None:
        """Register a new file type to be used for token instrumentation."""
        if not suffix.startswith('.'):
            suffix = f'.{suffix}'
        COMMENT_CHARS[suffix] = comment_chars

    @property
    def placeholders(self) -> List[Placeholder]:
        return self.instantiator.placeholders

    def __call__(self, **kwargs: Any) -> Any:
        with self.instantiator.instantiate(**kwargs) as folder:
            if self.verbose:
                print(f'Running {self.command} from {folder.parent} which holds {folder}')
            output: str = utils.CommandFunction(self.command, cwd=folder.parent)()
        if self.verbose:
            print(f'FolderFunction recovered full output:\n{output}')
        self.last_full_output = output.strip()
        if not output:
            raise ValueError('No output')
        for postproc in self.postprocessings:
            output = postproc(output)
        if self.verbose:
            print(f'FolderFunction returns: {output}')
        return output


def get_last_line_as_float(output: str) -> float:
    split_output: List[str] = output.strip().splitlines()
    return float(split_output[-1])
