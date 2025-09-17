import os
import pycodestyle as pep8
from typing import Any, Dict, List, Optional
from linting import linter


class Pep8Error(linter.LintError):
    """PEP-8 linting error class"""

    def __init__(self, filename: str, loc: int, offset: int, code: str, text: str, level: str = 'E') -> None:
        ct_tuple: tuple[str, str] = (code, text)
        err_str: str = '[{0}] PEP 8 (%s): %s'.format(level)
        super(Pep8Error, self).__init__(filename, loc, level, err_str, ct_tuple, offset=offset, text=text)


class Pep8Warning(linter.LintError):
    """PEP-8 lintng warning class"""

    def __init__(self, filename: str, loc: int, offset: int, code: str, text: str, level: str = 'W') -> None:
        ct_tuple: tuple[str, str] = (code, text)
        err_str: str = '[{0}] PEP 8 (%s): %s'.format(level)
        super(Pep8Warning, self).__init__(filename, loc, level, err_str, ct_tuple, offset=offset, text=text)


class Pep8Linter(linter.Linter):
    """Linter for pep8 Linter"""

    def lint(self, settings: Dict[str, Any], code: str, filename: str) -> List[Dict[str, Any]]:
        """Run the pep8 code checker with the given options"""
        errors: List[linter.LintError] = []
        check_params: Dict[str, Any] = {
            'ignore': settings.get('pep8_ignore', []),
            'max_line_length': settings.get('pep8_max_line_length', pep8.MAX_LINE_LENGTH),
            'levels': settings.get('pep8_error_levels', {'E': 'W', 'W': 'V', 'V': 'V'})
        }
        errors.extend(self.check(code, filename, settings.get('pep8_rcfile'), **check_params))
        return self.parse(errors)

    def check(
        self,
        code: str,
        filename: str,
        rcfile: Optional[str],
        ignore: List[str],
        max_line_length: int,
        levels: Dict[str, str]
    ) -> List[linter.LintError]:
        """Check the code with pyflakes to find errors"""
        messages: List[linter.LintError] = []
        _lines: List[str] = code.split('\n')
        if _lines:

            class AnacondaReport(pep8.BaseReport):
                """Helper class to report PEP8 problems"""

                def error(self, line_number: int, offset: int, text: str, check: Any) -> Optional[str]:
                    """Report an error, according to options"""
                    col: int = line_number
                    err_code: str = text[:4]
                    message_text: str = text[5:]
                    if self._ignore_code(err_code):
                        return None
                    if err_code in self.counters:
                        self.counters[err_code] += 1
                    else:
                        self.counters[err_code] = 1
                        self.messages[err_code] = message_text
                    if err_code in self.expected:
                        return None
                    self.file_errors += 1
                    self.total_errors += 1
                    pep8_error: bool = err_code.startswith('E')
                    klass = Pep8Error if pep8_error else Pep8Warning
                    messages.append(klass(filename, col, offset, err_code, message_text, levels[err_code[0]]))
                    return err_code

            params: Dict[str, Any] = {'reporter': AnacondaReport}
            if not rcfile:
                _ignore: List[str] = ignore
                params['ignore'] = _ignore
            else:
                params['config_file'] = os.path.expanduser(rcfile)
            options = pep8.StyleGuide(**params).options
            if not rcfile:
                options.max_line_length = max_line_length
            good_lines: List[str] = [l + '\n' for l in _lines]
            if good_lines:
                good_lines[-1] = good_lines[-1].rstrip('\n')
            if good_lines and not good_lines[-1]:
                good_lines = good_lines[:-1]
            pep8.Checker(filename, good_lines, options=options).check_all()
        return messages

    def parse(self, errors: List[linter.LintError]) -> List[Dict[str, Any]]:
        errors_list: List[Dict[str, Any]] = []
        if errors is None:
            return errors_list
        self.sort_errors(errors)
        for error in errors:
            error_level: str = self.prepare_error_level(error)
            message: str = error.message.capitalize()
            offset: int = error.offset
            error_data: Dict[str, Any] = {
                'underline_range': True,
                'level': error_level,
                'lineno': error.lineno,
                'offset': offset,
                'message': message,
                'raw_error': str(error)
            }
            errors_list.append(error_data)
        return errors_list