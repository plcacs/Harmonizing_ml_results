#!/usr/bin/env python3
"""
Analyze docstrings to detect errors.

If no argument is provided, it does a quick check of docstrings and returns
a csv with all API functions and results of basic checks.

If a function or method is provided in the form "pandas.function",
"pandas.module.class.method", etc. a list of all errors in the docstring for
the specified function or method.

Usage::
    $ ./validate_docstrings.py
    $ ./validate_docstrings.py pandas.DataFrame.head
"""
from __future__ import annotations

import argparse
import collections
from collections import defaultdict
import doctest
import importlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Set, Tuple, TextIO

from numpydoc.docscrape import get_doc_object
from numpydoc.validate import ERROR_MSGS as NUMPYDOC_ERROR_MSGS, Validator, validate

matplotlib.use('template')
IGNORE_VALIDATION: Set[str] = {
    'Styler.env', 'Styler.template_html', 'Styler.template_html_style', 'Styler.template_html_table',
    'Styler.template_latex', 'Styler.template_typst', 'Styler.template_string', 'Styler.loader',
    'errors.InvalidComparison', 'errors.LossySetitemError', 'errors.NoBufferPresent', 'errors.IncompatibilityWarning',
    'errors.PyperclipException', 'errors.PyperclipWindowsException'
}
PRIVATE_CLASSES: List[str] = ['NDFrame', 'IndexOpsMixin']
ERROR_MSGS: Dict[str, str] = {
    'GL04': 'Private classes ({mentioned_private_classes}) should not be mentioned in public docstrings',
    'PD01': "Use 'array-like' rather than 'array_like' in docstrings.",
    'SA05': '{reference_name} in `See Also` section does not need `pandas` prefix, use {right_reference} instead.',
    'EX03': 'flake8 error: line {line_number}, col {col_number}: {error_code} {error_message}',
    'EX04': 'Do not import {imported_library}, as it is imported automatically for the examples (numpy as np, pandas as pd)'
}
ALL_ERRORS: Set[str] = set(NUMPYDOC_ERROR_MSGS).union(set(ERROR_MSGS))
duplicated_errors: Set[str] = set(NUMPYDOC_ERROR_MSGS).intersection(set(ERROR_MSGS))
assert not duplicated_errors, f'Errors {duplicated_errors} exist in both pandas and numpydoc, should they be removed from pandas?'

def pandas_error(code: str, **kwargs: Any) -> Tuple[str, str]:
    """
    Copy of the numpydoc error function, since ERROR_MSGS can't be updated
    with our custom errors yet.
    """
    return (code, ERROR_MSGS[code].format(**kwargs))

def get_api_items(api_doc_fd: TextIO) -> Iterator[Tuple[str, Any, str, str]]:
    """
    Yield information about all public API items.

    Parse api.rst file from the documentation, and extract all the functions,
    methods, classes, attributes... This should include all pandas public API.

    Parameters
    ----------
    api_doc_fd : file descriptor
        A file descriptor of the API documentation page, containing the table
        of contents with all the public API.

    Yields
    ------
    name : str
        The name of the object (e.g. 'pandas.Series.str.upper').
    func : function
        The object itself. In most cases this will be a function or method,
        but it can also be classes, properties, cython objects...
    section : str
        The name of the section in the API page where the object item is
        located.
    subsection : str
        The name of the subsection in the API page where the object item is
        located.
    """
    current_module: str = 'pandas'
    previous_line: str = ''
    current_section: str = ''
    current_subsection: str = ''
    position: Optional[str] = None
    for line in api_doc_fd:
        line_stripped: str = line.strip()
        if len(line_stripped) == len(previous_line):
            if set(line_stripped) == set('-'):
                current_section = previous_line
                continue
            if set(line_stripped) == set('~'):
                current_subsection = previous_line
                continue
        if line_stripped.startswith('.. currentmodule::'):
            current_module = line_stripped.replace('.. currentmodule::', '').strip()
            continue
        if line_stripped == '.. autosummary::':
            position = 'autosummary'
            continue
        if position == 'autosummary':
            if line_stripped == '':
                position = 'items'
                continue
        if position == 'items':
            if line_stripped == '':
                position = None
                continue
            if line_stripped in IGNORE_VALIDATION:
                continue
            func: Any = importlib.import_module(current_module)
            for part in line_stripped.split('.'):
                func = getattr(func, part)
            yield (f'{current_module}.{line_stripped}', func, current_section, current_subsection)
        previous_line = line_stripped

class PandasDocstring(Validator):
    def __init__(self, func_name: str, doc_obj: Optional[Any] = None) -> None:
        self.func_name: str = func_name
        if doc_obj is None:
            doc_obj = get_doc_object(Validator._load_obj(func_name))
        super().__init__(doc_obj)

    @property
    def name(self) -> str:
        return self.func_name

    @property
    def mentioned_private_classes(self) -> List[str]:
        return [klass for klass in PRIVATE_CLASSES if klass in self.raw_doc]

    @property
    def examples_source_code(self) -> List[str]:
        lines = doctest.DocTestParser().get_examples(self.raw_doc)
        return [line.source for line in lines]

    def validate_pep8(self) -> Iterator[Tuple[str, str, int, int]]:
        if not self.examples:
            return
        content: str = ''.join(
            ('import numpy as np  # noqa: F401\n',
             'import pandas as pd  # noqa: F401\n',
             *self.examples_source_code))
        error_messages: List[str] = []
        file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
        try:
            file.write(content)
            file.flush()
            cmd = [
                sys.executable, '-m', 'flake8',
                '--format=%(row)d\t%(col)d\t%(code)s\t%(text)s',
                '--max-line-length=88',
                '--ignore=E203,E3,W503,W504,E402,E731,E128,E124,E704',
                file.name
            ]
            response = subprocess.run(cmd, capture_output=True, check=False, text=True)
            for output in ('stdout', 'stderr'):
                out: str = getattr(response, output)
                out = out.replace(file.name, '')
                messages = out.strip('\n').splitlines()
                if messages:
                    error_messages.extend(messages)
        finally:
            file.close()
            os.unlink(file.name)
        for error_message in error_messages:
            parts = error_message.split('\t', maxsplit=3)
            if len(parts) < 4:
                continue
            line_number, col_number, error_code, message = parts
            yield (error_code, message, int(line_number) - 2, int(col_number))

    def non_hyphenated_array_like(self) -> bool:
        return 'array_like' in self.raw_doc

def pandas_validate(func_name: str) -> Dict[str, Any]:
    """
    Call the numpydoc validation, and add the errors specific to pandas.

    Parameters
    ----------
    func_name : str
        Name of the object of the docstring to validate.

    Returns
    -------
    dict
        Information about the docstring and the errors found.
    """
    func_obj: Any = Validator._load_obj(func_name)
    doc_obj: Any = get_doc_object(func_obj, doc=func_obj.__doc__)
    doc: PandasDocstring = PandasDocstring(func_name, doc_obj)
    result: Dict[str, Any] = validate(doc_obj)
    mentioned_errs: List[str] = doc.mentioned_private_classes
    if mentioned_errs:
        result['errors'].append(pandas_error('GL04', mentioned_private_classes=', '.join(mentioned_errs)))
    if doc.see_also:
        result['errors'].extend(
            (pandas_error('SA05', reference_name=rel_name, right_reference=rel_name[len('pandas.'):])
             for rel_name in doc.see_also if rel_name.startswith('pandas.'))
        )
    result['examples_errs'] = ''
    if doc.examples:
        for error_code, error_message, line_number, col_number in doc.validate_pep8():
            result['errors'].append(pandas_error('EX03', error_code=error_code, error_message=error_message, line_number=line_number, col_number=col_number))
        examples_source_code: str = ''.join(doc.examples_source_code)
        result['errors'].extend(
            (pandas_error('EX04', imported_library=wrong_import)
             for wrong_import in ('numpy', 'pandas')
             if f'import {wrong_import}' in examples_source_code)
        )
    if doc.non_hyphenated_array_like():
        result['errors'].append(pandas_error('PD01'))
    plt.close('all')
    return result

def validate_all(prefix: Optional[str], ignore_deprecated: bool = False) -> Dict[str, Any]:
    """
    Execute the validation of all docstrings, and return a dict with the
    results.

    Parameters
    ----------
    prefix : str or None
        If provided, only the docstrings that start with this pattern will be
        validated. If None, all docstrings will be validated.
    ignore_deprecated: bool, default False
        If True, deprecated objects are ignored when validating docstrings.

    Returns
    -------
    dict
        A dictionary with an item for every function/method... containing
        all the validation information.
    """
    result: Dict[str, Any] = {}
    seen: Dict[Tuple[Any, Any], str] = {}
    for func_name, _, section, subsection in get_all_api_items():
        if prefix and (not func_name.startswith(prefix)):
            continue
        doc_info: Dict[str, Any] = pandas_validate(func_name)
        if ignore_deprecated and doc_info.get('deprecated'):
            continue
        result[func_name] = doc_info
        shared_code_key = (doc_info['file'], doc_info['file_line'])
        shared_code = seen.get(shared_code_key, '')
        result[func_name].update({'in_api': True, 'section': section, 'subsection': subsection, 'shared_code_with': shared_code})
        seen[shared_code_key] = func_name
    return result

def get_all_api_items() -> Iterator[Tuple[str, Any, str, str]]:
    base_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    api_doc_fnames: pathlib.Path = pathlib.Path(base_path, 'doc', 'source', 'reference')
    for api_doc_fname in api_doc_fnames.glob('*.rst'):
        with open(api_doc_fname, encoding='utf-8') as f:
            yield from get_api_items(f)

def print_validate_all_results(output_format: str, prefix: Optional[str], ignore_deprecated: bool, ignore_errors: Optional[Dict[Optional[str], Set[str]]]) -> int:
    if output_format not in ('default', 'json', 'actions'):
        raise ValueError(f'Unknown output_format "{output_format}"')
    if ignore_errors is None:
        ignore_errors = {}
    result: Dict[str, Any] = validate_all(prefix, ignore_deprecated)
    if output_format == 'json':
        sys.stdout.write(json.dumps(result))
        return 0
    out_prefix: str = '##[error]' if output_format == 'actions' else ''
    exit_status: int = 0
    for func_name, res in result.items():
        error_messages: Dict[str, str] = dict(res['errors'])
        actual_failures: Set[str] = set(error_messages)
        expected_failures: Set[str] = ignore_errors.get(func_name, set()) | ignore_errors.get(None, set())
        for err_code in actual_failures - expected_failures:
            sys.stdout.write(f"{out_prefix}{res['file']}:{res['file_line']}:{err_code}:{func_name}:{error_messages[err_code]}\n")
            exit_status += 1
        for err_code in ignore_errors.get(func_name, set()) - actual_failures:
            sys.stdout.write(f"{out_prefix}{res['file']}:{res['file_line']}:{err_code}:{func_name}:EXPECTED TO FAIL, BUT NOT FAILING\n")
            exit_status += 1
    return exit_status

def print_validate_one_results(func_name: str, ignore_errors: Dict[Optional[str], Set[str]]) -> int:
    def header(title: str, width: int = 80, char: str = '#') -> str:
        full_line: str = char * width
        side_len: int = (width - len(title) - 2) // 2
        adj: str = '' if len(title) % 2 == 0 else ' '
        title_line: str = f'{char * side_len} {title}{adj} {char * side_len}'
        return f'\n{full_line}\n{title_line}\n{full_line}\n\n'
    result: Dict[str, Any] = pandas_validate(func_name)
    result['errors'] = [(code, message) for code, message in result['errors'] if code not in ignore_errors.get(None, set())]
    sys.stderr.write(header(f'Docstring ({func_name})'))
    sys.stderr.write(f"{result['docstring']}\n")
    sys.stderr.write(header('Validation'))
    if result['errors']:
        sys.stderr.write(f"{len(result['errors'])} Errors found for `{func_name}`:\n")
        for err_code, err_desc in result['errors']:
            sys.stderr.write(f"\t{err_code}\t{err_desc}\n")
    else:
        sys.stderr.write(f'Docstring for "{func_name}" correct. :)\n')
    if result['examples_errs']:
        sys.stderr.write(header('Doctests'))
        sys.stderr.write(result['examples_errs'])
    return len(result['errors']) + (len(result['examples_errs']) if isinstance(result['examples_errs'], str) and result['examples_errs'] else 0)

def _format_ignore_errors(raw_ignore_errors: Optional[List[str]]) -> DefaultDict[Optional[str], Set[str]]:
    ignore_errors: DefaultDict[Optional[str], Set[str]] = defaultdict(set)
    if raw_ignore_errors:
        for error_codes in raw_ignore_errors:
            obj_name: Optional[str] = None
            if ' ' in error_codes:
                obj_name, error_codes = error_codes.split(' ', 1)
            if obj_name:
                if obj_name in ignore_errors:
                    raise ValueError(f'Object `{obj_name}` is present in more than one --ignore_errors argument. Please use it once and specify the errors separated by commas.')
                ignore_errors[obj_name] = set(error_codes.split(','))
                unknown_errors: Set[str] = ignore_errors[obj_name] - ALL_ERRORS
                if unknown_errors:
                    raise ValueError(f'Object `{obj_name}` is ignoring errors {unknown_errors} which are not known. Known errors are: {ALL_ERRORS}')
            else:
                ignore_errors[None].update(set(error_codes.split(',')))
        unknown_errors: Set[str] = ignore_errors.get('*', set()) - ALL_ERRORS
        if unknown_errors:
            raise ValueError(f'Unknown errors {unknown_errors} specified using --ignore_errors Known errors are: {ALL_ERRORS}')
    return ignore_errors

def main(func_name: Optional[str],
         output_format: str,
         prefix: Optional[str],
         ignore_deprecated: bool,
         ignore_errors: DefaultDict[Optional[str], Set[str]]) -> int:
    """
    Main entry point. Call the validation for one or for all docstrings.
    """
    if func_name is None:
        return print_validate_all_results(output_format, prefix, ignore_deprecated, ignore_errors)
    else:
        return print_validate_one_results(func_name, ignore_errors)

if __name__ == '__main__':
    format_opts: Tuple[str, ...] = ('default', 'json', 'actions')
    func_help: str = 'function or method to validate (e.g. pandas.DataFrame.head) if not provided, all docstrings are validated and returned as JSON'
    argparser = argparse.ArgumentParser(description='validate pandas docstrings')
    argparser.add_argument('function', nargs='?', default=None, help=func_help)
    argparser.add_argument('--format', default='default', choices=format_opts,
                           help=f'format of the output when validating multiple docstrings (ignored when validating one). It can be {str(format_opts)[1:-1]}')
    argparser.add_argument('--prefix', default=None,
                           help='pattern for the docstring names, in order to decide which ones will be validated. A prefix "pandas.Series.str." will make the script validate all the docstrings of methods starting by this pattern. It is ignored if parameter function is provided')
    argparser.add_argument('--ignore_deprecated', default=False, action='store_true',
                           help='if this flag is set, deprecated objects are ignored when validating all docstrings')
    argparser.add_argument('--ignore_errors', '-i', default=None, action='append',
                           help="comma-separated list of error codes (e.g. 'PR02,SA01'), with optional object path to ignore errors for a single object (e.g. pandas.DataFrame.head PR02,SA01). Partial validation for more than one function can be achieved by repeating this parameter.")
    args = argparser.parse_args(sys.argv[1:])
    sys.exit(main(args.function, args.format, args.prefix, args.ignore_deprecated, _format_ignore_errors(args.ignore_errors)))