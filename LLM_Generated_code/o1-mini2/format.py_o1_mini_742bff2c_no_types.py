import re
import sys
from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import Optional, TextIO
from re import Pattern
try:
    import colorama
except ImportError:
    colorama_unavailable: bool = True
else:
    colorama_unavailable = False
ADDED_LINE_PATTERN: Pattern[str] = re.compile('\\+[^+]')
REMOVED_LINE_PATTERN: Pattern[str] = re.compile('-[^-]')


def format_simplified(import_line):
    import_line = import_line.strip()
    if import_line.startswith('from '):
        import_line = import_line.replace('from ', '')
        import_line = import_line.replace(' import ', '.')
    elif import_line.startswith('import '):
        import_line = import_line.replace('import ', '')
    return import_line


def format_natural(import_line):
    import_line = import_line.strip()
    if not import_line.startswith('from ') and not import_line.startswith(
        'import '):
        if '.' not in import_line:
            return f'import {import_line}'
        parts = import_line.split('.')
        end = parts.pop(-1)
        return f"from {'.'.join(parts)} import {end}"
    return import_line


def show_unified_diff(*, file_input: str, file_output: str, file_path:
    Optional[Path], output: Optional[TextIO]=None, color_output: bool=False):
    """Shows a unified_diff for the provided input and output against the provided file path.

    - **file_input**: A string that represents the contents of a file before changes.
    - **file_output**: A string that represents the contents of a file after changes.
    - **file_path**: A Path object that represents the file path of the file being changed.
    - **output**: A stream to output the diff to. If non is provided uses sys.stdout.
    - **color_output**: Use color in output if True.
    """
    printer: BasicPrinter = create_terminal_printer(color_output, output)
    file_name: str = '' if file_path is None else str(file_path)
    file_mtime: str = str(datetime.now() if file_path is None else datetime
        .fromtimestamp(file_path.stat().st_mtime))
    unified_diff_lines = unified_diff(file_input.splitlines(keepends=True),
        file_output.splitlines(keepends=True), fromfile=file_name +
        ':before', tofile=file_name + ':after', fromfiledate=file_mtime,
        tofiledate=str(datetime.now()))
    for line in unified_diff_lines:
        printer.diff_line(line)


def ask_whether_to_apply_changes_to_file(file_path):
    answer: Optional[str] = None
    while answer not in ('yes', 'y', 'no', 'n', 'quit', 'q'):
        answer = input(f"Apply suggested changes to '{file_path}' [y/n/q]? ")
        answer = answer.lower()
        if answer in ('no', 'n'):
            return False
        if answer in ('quit', 'q'):
            sys.exit(1)
    return True


def remove_whitespace(content, line_separator='\n'):
    content = content.replace(line_separator, '').replace(' ', '').replace(
        '\x0c', '')
    return content


class BasicPrinter:
    ERROR: str = 'ERROR'
    SUCCESS: str = 'SUCCESS'

    def __init__(self, error, success, output=None):
        self.output: TextIO = output or sys.stdout
        self.success_message: str = success
        self.error_message: str = error

    def success(self, message):
        print(self.success_message.format(success=self.SUCCESS, message=
            message), file=self.output)

    def error(self, message):
        print(self.error_message.format(error=self.ERROR, message=message),
            file=sys.stderr)

    def diff_line(self, line):
        self.output.write(line)


class ColoramaPrinter(BasicPrinter):
    ADDED_LINE: str
    REMOVED_LINE: str
    ERROR: str
    SUCCESS: str

    def __init__(self, error, success, output):
        super().__init__(error, success, output=output)
        self.ERROR = self.style_text('ERROR', colorama.Fore.RED)
        self.SUCCESS = self.style_text('SUCCESS', colorama.Fore.GREEN)
        self.ADDED_LINE = colorama.Fore.GREEN
        self.REMOVED_LINE = colorama.Fore.RED

    @staticmethod
    def style_text(text, style=None):
        if style is None:
            return text
        return style + text + str(colorama.Style.RESET_ALL)

    def diff_line(self, line):
        style: Optional[str] = None
        if re.match(ADDED_LINE_PATTERN, line):
            style = self.ADDED_LINE
        elif re.match(REMOVED_LINE_PATTERN, line):
            style = self.REMOVED_LINE
        self.output.write(self.style_text(line, style))


def create_terminal_printer(color, output=None, error='', success=''):
    if color and colorama_unavailable:
        no_colorama_message: str = """
Sorry, but to use --color (color_output) the colorama python package is required.

Reference: https://pypi.org/project/colorama/

You can either install it separately on your system or as the colors extra for isort. Ex: 

$ pip install isort[colors]
"""
        print(no_colorama_message, file=sys.stderr)
        sys.exit(1)
    if not colorama_unavailable:
        colorama.init(strip=False)
    return ColoramaPrinter(error, success, output) if color else BasicPrinter(
        error, success, output)
