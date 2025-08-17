import io
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, IO

import pytest

from scripts import validate_docstrings


class BadDocstrings:
    """Everything here has a bad docstring"""

    def private_classes(self) -> None:
        """
        This mentions NDFrame, which is not correct.
        """

    def prefix_pandas(self) -> None:
        """
        Have `pandas` prefix in See Also section.

        See Also
        --------
        pandas.Series.rename : Alter Series index labels or name.
        DataFrame.head : The first `n` rows of the caller object.
        """

    def redundant_import(self, paramx: Optional[Any] = None, paramy: Optional[Any] = None) -> None:
        """
        A sample DataFrame method.

        Should not import numpy and pandas.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = pd.DataFrame(np.ones((3, 3)),
        ...                   columns=('a', 'b', 'c'))
        >>> df.all(axis=1)
        0    True
        1    True
        2    True
        dtype: bool
        >>> df.all(bool_only=True)
        Series([], dtype: bool)
        """

    def unused_import(self) -> None:
        """
        Examples
        --------
        >>> import pandas as pdf
        >>> df = pd.DataFrame(np.ones((3, 3)), columns=('a', 'b', 'c'))
        """

    def missing_whitespace_around_arithmetic_operator(self) -> None:
        """
        Examples
        --------
        >>> 2+5
        7
        """

    def indentation_is_not_a_multiple_of_four(self) -> None:
        """
        Examples
        --------
        >>> if 2 + 5:
        ...   pass
        """

    def missing_whitespace_after_comma(self) -> None:
        """
        Examples
        --------
        >>> df = pd.DataFrame(np.ones((3,3)),columns=('a','b', 'c'))
        """

    def write_array_like_with_hyphen_not_underscore(self) -> None:
        """
        In docstrings, use array-like over array_like
        """

    def leftover_files(self) -> None:
        """
        Examples
        --------
        >>> import pathlib
        >>> pathlib.Path("foo.txt").touch()
        """


class TestValidator:
    def _import_path(self, klass: Optional[str] = None, func: Optional[str] = None) -> str:
        """
        Build the required import path for tests in this module.

        Parameters
        ----------
        klass : str
            Class name of object in module.
        func : str
            Function name of object in module.

        Returns
        -------
        str
            Import path of specified object in this module
        """
        base_path = "scripts.tests.test_validate_docstrings"

        if klass:
            base_path = f"{base_path}.{klass}"

        if func:
            base_path = f"{base_path}.{func}"

        return base_path

    def test_bad_class(self, capsys: pytest.CaptureFixture[str]) -> None:
        errors = validate_docstrings.pandas_validate(
            self._import_path(klass="BadDocstrings")
        )["errors"]
        assert isinstance(errors, list)
        assert errors

    @pytest.mark.parametrize(
        "klass,func,msgs",
        [
            (
                "BadDocstrings",
                "private_classes",
                (
                    "Private classes (NDFrame) should not be mentioned in public "
                    "docstrings",
                ),
            ),
            (
                "BadDocstrings",
                "prefix_pandas",
                (
                    "pandas.Series.rename in `See Also` section "
                    "does not need `pandas` prefix",
                ),
            ),
            # Examples tests
            (
                "BadDocstrings",
                "redundant_import",
                ("Do not import numpy, as it is imported automatically",),
            ),
            (
                "BadDocstrings",
                "redundant_import",
                ("Do not import pandas, as it is imported automatically",),
            ),
            (
                "BadDocstrings",
                "unused_import",
                (
                    "flake8 error: line 1, col 1: F401 'pandas as pdf' "
                    "imported but unused",
                ),
            ),
            (
                "BadDocstrings",
                "missing_whitespace_around_arithmetic_operator",
                (
                    "flake8 error: line 1, col 2: "
                    "E226 missing whitespace around arithmetic operator",
                ),
            ),
            (
                "BadDocstrings",
                "indentation_is_not_a_multiple_of_four",
                # with flake8 3.9.0, the message ends with four spaces,
                #  whereas in earlier versions, it ended with "four"
                (
                    "flake8 error: line 2, col 3: E111 indentation is not a "
                    "multiple of 4",
                ),
            ),
            (
                "BadDocstrings",
                "missing_whitespace_after_comma",
                ("flake8 error: line 1, col 33: E231 missing whitespace after ','",),
            ),
            (
                "BadDocstrings",
                "write_array_like_with_hyphen_not_underscore",
                ("Use 'array-like' rather than 'array_like' in docstrings",),
            ),
        ],
    )
    def test_bad_docstrings(
        self, 
        capsys: pytest.CaptureFixture[str], 
        klass: str, 
        func: str, 
        msgs: Tuple[str, ...]
    ) -> None:
        result = validate_docstrings.pandas_validate(
            self._import_path(klass=klass, func=func)
        )
        for msg in msgs:
            assert msg in " ".join([err[1] for err in result["errors"]])

    def test_validate_all_ignore_deprecated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc"),
                ],
                "warnings": [],
                "examples_errors": "",
                "deprecated": True,
            },
        )
        result = validate_docstrings.validate_all(prefix=None, ignore_deprecated=True)
        assert len(result) == 0

    def test_validate_all_ignore_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            validate_docstrings,
            "pandas_validate",
            lambda func_name: {
                "docstring": "docstring1",
                "errors": [
                    ("ER01", "err desc"),
                    ("ER02", "err desc"),
                    ("ER03", "err desc")
                ],
                "warnings": [],
                "examples_errors": "",
                "deprecated": True,
                "file": "file1",
                "file_line": "file_line1"
            },
        )
        monkeypatch.setattr(
            validate_docstrings,
            "get_all_api_items",
            lambda: [
                (
                    "pandas.DataFrame.align",
                    "func",
                    "current_section",
                    "current_subsection",
                ),
                (
                    "pandas.Index.all",
                    "func",
                    "current_section",
                    "current_subsection",
                ),
            ],
        )

        exit_status = validate_docstrings.print_validate_all_results(
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={None: {"ER03"}},
        )
        # two functions * two not ignored errors
        assert exit_status == 2 * 2

        exit_status = validate_docstrings.print_validate_all_results(
            output_format="default",
            prefix=None,
            ignore_deprecated=False,
            ignore_errors={
                None: {"ER03"},
                "pandas.DataFrame.align": {"ER01"},
                # ignoring an error that is not requested should be of no effect
                "pandas.Index.all": {"ER03"}
            }
        )
        # two functions * two not global ignored errors - one function ignored error
        assert exit_status == 2 * 2 - 1