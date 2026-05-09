from __future__ import annotations
from collections import abc, defaultdict
import csv
from io import StringIO
import re
from typing import IO, TYPE_CHECKING, Any, DefaultDict, Literal, cast, final
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import EmptyDataError, ParserError, ParserWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import is_bool_dtype, is_extension_array_dtype, is_integer, is_numeric_dtype, is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype
from pandas.core.dtypes.inference import is_dict_like
from pandas.core import algorithms
from pandas.core.arrays import Categorical, ExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import Index
from pandas.io.common import dedup_names, is_potential_multi_index
from pandas.io.parsers.base_parser import ParserBase, evaluate_callable_usecols, get_na_values, parser_defaults, validate_parse_dates_presence

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping, Sequence
    from pandas._typing import ArrayLike, DtypeObj, ReadCsvBuffer, Scalar, T
    from pandas import MultiIndex, Series

_BOM: str = '\ufeff'

class PythonParser(ParserBase):
    def __init__(self, f: IO[str], **kwds: Any) -> None:
        """
        Workhorse function for processing nested list into DataFrame
        """
        super().__init__(kwds)
        self.data: list[str] = []
        self.buf: list[str] = []
        self.pos: int = 0
        self.line_pos: int = 0
        self.skiprows: Any = kwds['skiprows']
        if callable(self.skiprows):
            self.skipfunc: Callable[[int], bool] = self.skiprows
        else:
            self.skipfunc: Callable[[int], bool] = lambda x: x in self.skiprows
        self.skipfooter: int = _validate_skipfooter_arg(kwds['skipfooter'])
        self.delimiter: str = kwds['delimiter']
        self.quotechar: str = kwds['quotechar']
        if isinstance(self.quotechar, str):
            self.quotechar: str = str(self.quotechar)
        self.escapechar: str = kwds['escapechar']
        self.doublequote: bool = kwds['doublequote']
        self.skipinitialspace: bool = kwds['skipinitialspace']
        self.lineterminator: str = kwds['lineterminator']
        self.quoting: int = kwds['quoting']
        self.skip_blank_lines: bool = kwds['skip_blank_lines']
        self.has_index_names: bool = kwds.get('has_index_names', False)
        self.thousands: str = kwds['thousands']
        self.decimal: str = kwds['decimal']
        self.comment: str = kwds['comment']
        if isinstance(f, list):
            self.data: list[str] = f
        else:
            assert hasattr(f, 'readline')
            self.data: list[str] = self._make_reader(f)
        self._col_indices: list[int] | None = None
        columns, self.num_original_columns, self.unnamed_cols = self._infer_columns()
        self.columns, self.index_names, self.col_names, _ = self._extract_multi_indexer_columns(columns, self.index_names)
        self.orig_names: list[str] = list(self.columns)
        index_names, self.orig_names, self.columns = self._get_index_name()
        if self.index_names is None:
            self.index_names: list[str] | None = index_names
        if self._col_indices is None:
            self._col_indices: list[int] = list(range(len(self.columns)))
        self._no_thousands_columns: set[int] = self._set_no_thousand_columns()
        if len(self.decimal) != 1:
            raise ValueError('Only length-1 decimal markers supported')

    @cache_readonly
    def num(self) -> re.Pattern[str]:
        decimal: str = re.escape(self.decimal)
        if self.thousands is None:
            regex: str = f'^[\\-\\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\\-?[0-9]+)?$'
        else:
            thousands: str = re.escape(self.thousands)
            regex: str = f'^[\\-\\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?([0-9]?(E|e)\\-?[0-9]+)?$'
        return re.compile(regex)

    def _make_reader(self, f: IO[str]) -> abc.Iterator[str]:
        sep: str = self.delimiter
        if sep is None or len(sep) == 1:
            if self.lineterminator:
                raise ValueError('Custom line terminators not supported in python parser (yet)')

            class MyDialect(csv.Dialect):
                delimiter: str = self.delimiter
                quotechar: str = self.quotechar
                escapechar: str = self.escapechar
                doublequote: bool = self.doublequote
                skipinitialspace: bool = self.skipinitialspace
                quoting: int = self.quoting
                lineterminator: str = '\n'
            dia: csv.Dialect = MyDialect
            if sep is not None:
                dia.delimiter: str = sep
            else:
                line: str = f.readline()
                lines: list[str] = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line: str = f.readline()
                    lines: list[str] = self._check_comments([[line]])[0]
                lines_str: list[str] = cast(list[str], lines)
                line: str = lines_str[0]
                self.pos += 1
                self.line_pos += 1
                sniffed: csv.Sniffer = csv.Sniffer().sniff(line)
                dia.delimiter: str = sniffed.delimiter
                line_rdr: csv.reader = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))
            reader: csv.reader = csv.reader(f, dialect=dia, strict=True)
        else:

            def _read() -> abc.Iterator[str]:
                line: str = f.readline()
                pat: re.Pattern[str] = re.compile(sep)
                yield pat.split(line.strip())
                for line in f:
                    yield pat.split(line.strip())
            reader: abc.Iterator[str] = _read()
        return reader

    def read(self, rows: int | None = None) -> tuple[Index, list[str], Any]:
        try:
            content: list[str] = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content: list[str] = []
            else:
                self.close()
                raise
        self._first_chunk: bool = False
        columns: list[str] = list(self.orig_names)
        if not len(content):
            names: list[str] = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
            index, columns, col_dict: tuple[Index, list[str], Any] = self._get_empty_meta(names, self.dtype)
            conv_columns: list[str] = self._maybe_make_multi_index_columns(columns, self.col_names)
            return (index, conv_columns, col_dict)
        indexnamerow: str | None = None
        if self.has_index_names and sum((int(v == '' or v is None) for v in content[0])) == len(columns):
            indexnamerow: str | None = content[0]
            content: list[str] = content[1:]
        alldata: list[str] = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)
        conv_data: Any = self._convert_data(data)
        conv_data: Any = self._do_date_conversions(columns, conv_data)
        index, result_columns: tuple[Index, list[str]] = self._make_index(alldata, columns, indexnamerow)
        return (index, result_columns, conv_data)

    def _exclude_implicit_index(self, alldata: list[str]) -> tuple[dict[str, str], list[str]]:
        names: list[str] = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
        offset: int = 0
        if self._implicit_index:
            offset: int = len(self.index_col)
        len_alldata: int = len(alldata)
        self._check_data_length(names, alldata)
        return ({name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata}, names)

    def get_chunk(self, size: int | None = None) -> tuple[Index, list[str], Any]:
        if size is None:
            size: int = self.chunksize
        return self.read(rows=size)

    def _convert_data(self, data: dict[str, str]) -> Any:
        clean_conv: dict[str, Callable[[str], str]] = self._clean_mapping(self.converters)
        clean_dtypes: dict[str, pandas_dtype] = self._clean_mapping(self.dtype)
        clean_na_values: dict[str, str] = {}
        clean_na_fvalues: dict[str, str] = {}
        if isinstance(self.na_values, dict):
            for col in self.na_values:
                if col is not None:
                    na_value: str = self.na_values[col]
                    na_fvalue: str = self.na_fvalues[col]
                    if isinstance(col, int) and col not in self.orig_names:
                        col: str = self.orig_names[col]
                    clean_na_values[col]: str = na_value
                    clean_na_fvalues[col]: str = na_fvalue
        else:
            clean_na_values: dict[str, str] = self.na_values
            clean_na_fvalues: dict[str, str] = self.na_fvalues
        return self._convert_to_ndarrays(data, clean_na_values, clean_na_fvalues, clean_conv, clean_dtypes)

    @final
    def _convert_to_ndarrays(self, dct: dict[str, str], na_values: dict[str, str], na_fvalues: dict[str, str], converters: dict[str, Callable[[str], str]] | None, dtypes: dict[str, pandas_dtype] | None) -> Any:
        result: dict[str, str] = {}
        parse_date_cols: list[str] = validate_parse_dates_presence(self.parse_dates, self.columns)
        for c, values in dct.items():
            conv_f: Callable[[str], str] | None = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type: pandas_dtype | None = dtypes.get(c, None)
            else:
                cast_type: pandas_dtype | None = dtypes
            if self.na_filter:
                col_na_values, col_na_fvalues = get_na_values(c, na_values, na_fvalues, self.keep_default_na)
            else:
                col_na_values, col_na_fvalues = (set(), set())
            if c in parse_date_cols:
                mask: np.ndarray = np.isin(values, set(col_na_values) | col_na_fvalues)
                np.putmask(values, mask, np.nan)
                result[c]: str = values
                continue
            if conv_f is not None:
                if cast_type is not None:
                    warnings.warn(f'Both a converter and dtype were specified for column {c} - only the converter will be used.', ParserWarning, stacklevel=find_stack_level())
                try:
                    values: str = lib.map_infer(values, conv_f)
                except ValueError:
                    mask: np.ndarray = np.isin(values, list(na_values)).view(np.uint8)
                    values: str = lib.map_infer_mask(values, conv_f, mask)
                cvals, na_count: tuple[str, int] = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool=False)
            else:
                is_ea: bool = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype: bool = is_ea or is_string_dtype(cast_type)
                try_num_bool: bool = not (cast_type and is_str_or_ea_dtype)
                cvals, na_count: tuple[str, int] = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool)
                if cast_type is not None:
                    cast_type: pandas_dtype = pandas_dtype(cast_type)
                if cast_type and (cvals.dtype != cast_type or is_ea):
                    if not is_ea and na_count > 0:
                        if is_bool_dtype(cast_type):
                            raise ValueError(f'Bool column has NA values in column {c}')
                    cvals: str = self._cast_types(cvals, cast_type, c)
            result[c]: str = cvals
        return result

    @final
    def _cast_types(self, values: str, cast_type: pandas_dtype, column: str) -> str:
        """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray or ExtensionArray
        cast_type : np.dtype or ExtensionDtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray or ExtensionArray
        """
        if isinstance(cast_type, CategoricalDtype):
            known_cats: bool = cast_type.categories is not None
            if not is_object_dtype(values.dtype) and (not known_cats):
                values: str = lib.ensure_string_array(values, skipna=False, convert_na_value=False)
            cats: Index = Index(values).unique().dropna()
            values: Categorical = Categorical._from_inferred_categories(cats, cats.get_indexer(values), cast_type, true_values=self.true_values)
        elif isinstance(cast_type, ExtensionDtype):
            array_type: type[ExtensionArray] = cast_type.construct_array_type()
            try:
                if isinstance(cast_type, BooleanDtype):
                    values_str: list[str] = [str(val) for val in values]
                    return array_type._from_sequence_of_strings(values_str, dtype=cast_type, true_values=self.true_values, false_values=self.false_values, none_values=self.na_values)
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(f'Extension Array: {array_type} must implement _from_sequence_of_strings in order to be used in parser methods') from err
        elif isinstance(values, ExtensionArray):
            values: str = values.astype(cast_type, copy=False)
        elif issubclass(cast_type.type, str):
            values: str = lib.ensure_string_array(values, skipna=True, convert_na_value=False)
        else:
            try:
                values: str = astype_array(values, cast_type, copy=True)
            except ValueError as err:
                raise ValueError(f'Unable to convert column {column} to type {cast_type}') from err
        return values

    @cache_readonly
    def _have_mi_columns(self) -> bool:
        if self.header is None:
            return False
        header: str | list[str] | tuple[str, ...] | np.ndarray = self.header
        if isinstance(header, (list, tuple, np.ndarray)):
            return len(header) > 1
        else:
            return False

    def _infer_columns(self) -> tuple[list[str], int, set[str]]:
        names: list[str] = self.names
        num_original_columns: int = 0
        clear_buffer: bool = True
        unnamed_cols: set[str] = set()
        if self.header is not None:
            header: str | list[str] | tuple[str, ...] | np.ndarray = self.header
            have_mi_columns: bool = self._have_mi_columns
            if isinstance(header, (list, tuple, np.ndarray)):
                if have_mi_columns:
                    header: list[str] = list(header) + [header[-1] + 1]
            else:
                header: list[str] = [header]
            columns: list[str] = []
            for level, hr in enumerate(header):
                try:
                    line: str = self._buffered_line()
                    while self.line_pos <= hr:
                        line: str = self._next_line()
                except StopIteration as err:
                    if 0 < self.line_pos <= hr and (not have_mi_columns or hr != header[-1]):
                        joi: list[str] = list(map(str, header[:-1] if have_mi_columns else header))
                        msg: str = f'[{",".join(joi)}], len of {len(joi)}, '
                        raise ValueError(f'Passed header={msg}but only {self.line_pos} lines in file') from err
                    if have_mi_columns and hr > 0:
                        if clear_buffer:
                            self.buf.clear()
                        columns.append([None] * len(columns[-1]))
                        return (columns, num_original_columns, unnamed_cols)
                    if not self.names:
                        raise EmptyDataError('No columns to parse from file') from err
                    line: str = self.names[:]
                this_columns: list[str] = []
                this_unnamed_cols: list[int] = []
                for i, c in enumerate(line):
                    if c == '':
                        if have_mi_columns:
                            col_name: str = f'Unnamed: {i}_level_{level}'
                        else:
                            col_name: str = f'Unnamed: {i}'
                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)
                if not have_mi_columns:
                    counts: defaultdict[str, int] = defaultdict(int)
                    col_loop_order: list[int] = [i for i in range(len(this_columns)) if i not in this_unnamed_cols] + this_unnamed_cols
                    for i in col_loop_order:
                        col: str = this_columns[i]
                        old_col: str = col
                        cur_count: int = counts[col]
                        if cur_count > 0:
                            while cur_count > 0:
                                counts[old_col] = cur_count + 1
                                col: str = f'{old_col}.{cur_count}'
                                if col in this_columns:
                                    cur_count += 1
                                else:
                                    cur_count = counts[col]
                            if self.dtype is not None and is_dict_like(self.dtype) and (self.dtype.get(old_col) is not None) and (self.dtype.get(col) is None):
                                self.dtype.update({col: self.dtype.get(old_col)})
                        this_columns[i] = col
                        counts[col] = cur_count + 1
                elif have_mi_columns:
                    if hr == header[-1]:
                        lc: int = len(this_columns)
                        sic: list[int] | None = self.index_col
                        ic: int = len(sic) if sic is not None else 0
                        unnamed_count: int = len(this_unnamed_cols)
                        if lc != unnamed_count and lc - ic > unnamed_count or ic == 0:
                            clear_buffer: bool = False
                            this_columns: list[str] = [None] * lc
                            self.buf = [self.buf[-1]]
                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})
                if len(columns) == 1:
                    num_original_columns: int = len(this_columns)
            if clear_buffer:
                self.buf.clear()
            if names is not None:
                try:
                    first_line: str = self._next_line()
                except StopIteration:
                    first_line: str | None = None
                len_first_data_row: int = 0 if first_line is None else len(first_line)
                if len(names) > len(columns[0]) and len(names) > len_first_data_row:
                    raise ValueError('Number of passed names did not match number of header fields in the file')
                if len(columns) > 1:
                    raise TypeError('Cannot pass names with multi-index columns')
                if self.usecols is not None:
                    self._handle_usecols(columns, names, num_original_columns)
                else:
                    num_original_columns: int = len(names)
                if self._col_indices is not None and len(names) != len(self._col_indices):
                    columns: list[str] = [[names[i] for i in sorted(self._col_indices)]]
                else:
                    columns: list[str] = [names]
            else:
                columns: list[str] = self._handle_usecols(columns, columns[0], num_original_columns)
        else:
            ncols: int = len(self._header_line)
            num_original_columns: int = ncols
            if not names:
                columns: list[str] = [list(range(ncols))]
                columns: list[str] = self._handle_usecols(columns, columns[0], ncols)
            elif self.usecols is None or len(names) >= ncols:
                columns: list[str] = self._handle_usecols([names], names, ncols)
                num_original_columns: int = len(names)
            elif not callable(self.usecols) and len(names) != len(self.usecols):
                raise ValueError('Number of passed names did not match number of header fields in the file')
            else:
                columns: list[str] = [names]
                self._handle_usecols(columns, columns[0], ncols)
        return (columns, num_original_columns, unnamed_cols)

    @cache_readonly
    def _header_line(self) -> str | None:
        if self.header is not None:
            return None
        try:
            line: str = self._buffered_line()
        except StopIteration as err:
            if not self.names:
                raise EmptyDataError('No columns to parse from file') from err
            line: str = self.names[:]
        return line

    def _handle_usecols(self, columns: list[str], usecols_key: list[str], num_original_columns: int) -> list[str]:
        """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
        if self.usecols is not None:
            if callable(self.usecols):
                col_indices: list[int] = evaluate_callable_usecols(self.usecols, usecols_key)
            elif any((isinstance(u, str) for u in self.usecols)):
                if len(columns) > 1:
                    raise ValueError('If using multiple headers, usecols must be integers.')
                col_indices: list[int] = []
                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            self._validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                missing_usecols: list[int] = [col for col in self.usecols if col >= num_original_columns]
                if missing_usecols:
                    raise ParserError(f'Defining usecols with out-of-bounds indices is not allowed. {missing_usecols} are out-of-bounds.')
                col_indices: list[int] = self.usecols
            columns: list[str] = [[n for i, n in enumerate(column) if i in col_indices] for column in columns]
            self._col_indices: list[int] = sorted(col_indices)
        return columns

    def _buffered_line(self) -> str:
        """
        Return a line from buffer, filling buffer if required.
        """
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row: list[str]) -> list[str]:
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
        if not first_row:
            return first_row
        if not isinstance(first_row[0], str):
            return first_row
        if not first_row[0]:
            return first_row
        first_elt: str = first_row[0][0]
        if first_elt != _BOM:
            return first_row
        first_row_bom: str = first_row[0]
        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start: int = 2
            quote: str = first_row_bom[1]
            end: int = first_row_bom[2:].index(quote) + 2
            new_row: str = first_row_bom[start:end]
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1:]
        else:
            new_row: str = first_row_bom[1:]
        new_row_list: list[str] = [new_row]
        return new_row_list + first_row[1:]

    def _is_line_empty(self, line: str | list[str]) -> bool:
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
        return not line or all((not x for x in line))

    def _next_line(self) -> str:
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                if self.pos >= len(self.data):
                    break
                self.pos += 1
            while True:
                try:
                    line: str = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    if not self.skip_blank_lines and (self._is_line_empty(self.data[self.pos - 1]) or line):
                        break
                    if self.skip_blank_lines:
                        ret: list[str] = self._remove_empty_lines([line])
                        if ret:
                            line: str = ret[0]
                            break
                except IndexError as err:
                    raise StopIteration from err
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                next(self.data)
            while True:
                orig_line: str | None = self._next_iter_line(row_num=self.pos + 1)
                self.pos += 1
                if orig_line is not None:
                    line: str = self._check_comments([orig_line])[0]
                    if self.skip_blank_lines:
                        ret: list[str] = self._remove_empty_lines([line])
                        if ret:
                            line: str = ret[0]
                            break
                    elif self._is_line_empty(orig_line) or line:
                        break
        if self.pos == 1:
            line: str = self._check_for_bom(line)
        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg: str, row_num: int) -> None:
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg: str
            The error message to display.
        row_num: int
            The row number where the parsing error occurred.
            Because this row number is displayed, we 1-index,
            even though we 0-index internally.
        """
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        if self.on_bad_lines == self.BadLineHandleMethod.WARN:
            warnings.warn(f'Skipping line {row_num}: {msg}\n', ParserWarning, stacklevel=find_stack_level())

    def _next_iter_line(self, row_num: int) -> str | None:
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
        try:
            assert not isinstance(self.data, list)
            line: str = next(self.data)
            return line
        except csv.Error as e:
            if self.on_bad_lines in (self.BadLineHandleMethod.ERROR, self.BadLineHandleMethod.WARN):
                msg: str = str(e)
                if 'NULL byte' in msg or 'line contains NUL' in msg:
                    msg: str = "NULL byte detected. This byte cannot be processed in Python's native csv library at the moment, so please pass in engine='c' instead"
                if self.skipfooter > 0:
                    reason: str = "Error could possibly be due to parsing errors in the skipped footer rows (the skipfooter keyword is only applied after Python's csv library has parsed all rows)."
                    msg += '. ' + reason
                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines: list[str]) -> list[str]:
        if self.comment is None:
            return lines
        ret: list[str] = []
        for line in lines:
            rl: list[str] = []
            for x in line:
                if not isinstance(x, str) or self.comment not in x or x in self.na_values:
                    rl.append(x)
                else:
                    x: str = x[:x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines: list[str]) -> list[str]:
        """
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : list of list of Scalars
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : list of list of Scalars
            The same array of lines with the "empty" ones removed.
        """
        ret: list[str] = [line for line in lines if len(line) > 1 or (len(line) == 1 and (not isinstance(line[0], str) or line[0].strip()))]
        return ret

    def _check_thousands(self, lines: list[str]) -> list[str]:
        if self.thousands is None:
            return lines
        return self._search_replace_num_columns(lines=lines, search=self.thousands, replace='')

    def _search_replace_num_columns(self, lines: list[str], search: str, replace: str) -> list[str]:
        ret: list[str] = []
        for line in lines:
            rl: list[str] = []
            for i, x in enumerate(line):
                if not isinstance(x, str) or search not in x or i in self._no_thousands_columns or (not self.num.search(x.strip())):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines: list[str]) -> list[str]:
        if self.decimal == parser_defaults['decimal']:
            return lines
        return self._search_replace_num_columns(lines=lines, search=self.decimal, replace='.')

    def _get_index_name(self) -> tuple[list[str] | None, list[str], list[str]]:
        """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
        columns: list[str] = self.orig_names
        orig_names: list[str] = list(columns)
        columns: list[str] = list(columns)
        if self._header_line is not None:
            line: str | None = self._header_line
        else:
            try:
                line: str | None = self._next_line()
            except StopIteration:
                line: str | None = None
        try:
            next_line: str | None = self._next_line()
        except StopIteration:
            next_line: str | None = None
        implicit_first_cols: int = 0
        if line is not None:
            index_col: list[int] | bool | None = self.index_col
            if index_col is not False:
                implicit_first_cols: int = len(line) - self.num_original_columns
            if next_line is not None and self.header is not None and (index_col is not False):
                if len(next_line) == len(line) + self.num_original_columns:
                    self.index_col: list[int] = list(range(len(line)))
                    self.buf = self.buf[1:]
                    for c in reversed(line):
                        columns.insert(0, c)
                    orig_names: list[str] = list(columns)
                    self.num_original_columns: int = len(columns)
                    return (line, orig_names, columns)
        if implicit_first_cols > 0:
            self._implicit_index: bool = True
            if self.index_col is None:
                self.index_col: list[int] = list(range(implicit_first_cols))
            index_name: list[str] | None = None
        else:
            index_name, _, self.index_col: tuple[list[str] | None, list[int], list[int]] = self._clean_index_names(columns, self.index_col)
        return (index_name, orig_names, columns)

    def _rows_to_cols(self, content: list[str]) -> list[str]:
        col_len: int = self.num_original_columns
        if self._implicit_index:
            col_len += len(self.index_col)
        max_len: int = max((len(row) for row in content))
        if max_len > col_len and self.index_col is not False and (self.usecols is None):
            footers: int = self.skipfooter if self.skipfooter else 0
            bad_lines: list[tuple[int, int]] = []
            iter_content: abc.Iterator[tuple[int, str]] = enumerate(content)
            content_len: int = len(content)
            content: list[str] = []
            for i, _content in iter_content:
                actual_len: int = len(_content)
                if actual_len > col_len:
                    if callable(self.on_bad_lines):
                        new_l: str | None = self.on_bad_lines(_content)
                        if new_l is not None:
                            content.append(new_l)
                    elif self.on_bad_lines in (self.BadLineHandleMethod.ERROR, self.BadLineHandleMethod.WARN):
                        row_num: int = self.pos - (content_len - i + footers)
                        bad_lines.append((row_num, actual_len))
                        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                            break
                else:
                    content.append(_content)
            for row_num, actual_len in bad_lines:
                msg: str = f'Expected {col_len} fields in line {row_num + 1}, saw {actual_len}'
                if self.delimiter and len(self.delimiter) > 1 and (self.quoting != csv.QUOTE_NONE):
                    reason: str = 'Error could possibly be due to quotes being ignored when a multi-char delimiter is used.'
                    msg += '. ' + reason
                self._alert_malformed(msg, row_num + 1)
        zipped_content: list[str] = list(lib.to_object_array(content, min_width=col_len).T)
        if self.usecols:
            assert self._col_indices is not None
            col_indices: list[int] = self._col_indices
            if self._implicit_index:
                zipped_content: list[str] = [a for i, a in enumerate(zipped_content) if i < len(self.index_col) or i - len(self.index_col) in col_indices]
            else:
                zipped_content: list[str] = [a for i, a in enumerate(zipped_content) if i in col_indices]
        return zipped_content

    def _get_lines(self, rows: int | None) -> list[str]:
        lines: list[str] = self.buf
        new_rows: list[str] | None = None
        if rows is not None:
            if len(self.buf) >= rows:
                new_rows, self.buf = (self.buf[:rows], self.buf[rows:])
            else:
                rows -= len(self.buf)
        if new_rows is None:
            if isinstance(self.data, list):
                if self.pos > len(self.data):
                    raise StopIteration
                if rows is None:
                    new_rows: list[str] = self.data[self.pos:]
                    new_pos: int = len(self.data)
                else:
                    new_rows: list[str] = self.data[self.pos:self.pos + rows]
                    new_pos: int = self.pos + rows
                new_rows: list[str] = self._remove_skipped_rows(new_rows)
                lines.extend(new_rows)
                self.pos = new_pos
            else:
                new_rows: list[str] = []
                try:
                    if rows is not None:
                        row_index: int = 0
                        row_ct: int = 0
                        offset: int = self.pos if self.pos is not None else 0
                        while row_ct < rows:
                            new_row: str = next(self.data)
                            if not self.skipfunc(offset + row_index):
                                row_ct += 1
                            row_index += 1
                            new_rows.append(new_row)
                        len_new_rows: int = len(new_rows)
                        new_rows: list[str] = self._remove_skipped_rows(new_rows)
                        lines.extend(new_rows)
                    else:
                        rows: int = 0
                        while True:
                            next_row: str | None = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1
                            if next_row is not None:
                                new_rows.append(next_row)
                        len_new_rows: int = len(new_rows)
                except StopIteration:
                    len_new_rows: int = len(new_rows)
                    new_rows: list[str] = self._remove_skipped_rows(new_rows)
                    lines.extend(new_rows)
                    if len(lines) == 0:
                        raise
                self.pos += len_new_rows
            self.buf: list[str] = []
        else:
            lines: list[str] = new_rows
        if self.skipfooter:
            lines: list[str] = lines[:-self.skipfooter]
        lines: list[str] = self._check_comments(lines)
        if self.skip_blank_lines:
            lines: list[str] = self._remove_empty_lines(lines)
        lines: list[str] = self._check_thousands(lines)
        return self._check_decimal(lines)

    def _remove_skipped_rows(self, new_rows: list[str]) -> list[str]:
        if self.skiprows:
            return [row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)]
        return new_rows

    def _set_no_thousand_columns(self) -> set[int]:
        no_thousands_columns: set[int] = set()
        if self.columns and self.parse_dates:
            assert self._col_indices is not None
            no_thousands_columns: set[int] = self._set_noconvert_dtype_columns(self._col_indices, self.columns)
        if self.columns and self.dtype:
            assert self._col_indices is not None
            for i, col in zip(self._col_indices, self.columns):
                if not isinstance(self.dtype, dict) and (not is_numeric_dtype(self.dtype)):
                    no_thousands_columns.add(i)
                if isinstance(self.dtype, dict) and col in self.dtype and (not is_numeric_dtype(self.dtype[col]) or is_bool_dtype(self.dtype[col])):
                    no_thousands_columns.add(i)
        return no_thousands_columns

class FixedWidthReader(abc.Iterator[str]):
    """
    A reader of fixed-width lines.
    """

    def __init__(self, f: IO[str], colspecs: list[tuple[int, int]] | str, delimiter: str, comment: str | None, skiprows: set[int] | None, infer_nrows: int = 100) -> None:
        self.f: IO[str] = f
        self.buffer: abc.Iterator[str] | None = None
        self.delimiter: str = '\r\n' + delimiter if delimiter else '\n\r\t '
        self.comment: str | None = comment
        if colspecs == 'infer':
            self.colspecs: list[tuple[int, int]] = self.detect_colspecs(infer_nrows=infer_nrows, skiprows=skiprows)
        else:
            self.colspecs: list[tuple[int, int]] = colspecs
        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(f'column specifications must be a list or tuple, input was a {type(colspecs).__name__}')
        for colspec in self.colspecs:
            if not (isinstance(colspec, (tuple, list)) and len(colspec) == 2 and isinstance(colspec[0], (int, np.integer, type(None))) and isinstance(colspec[1], (int, np.integer, type(None)))):
                raise TypeError('Each column specification must be 2 element tuple or list of integers')

    def get_rows(self, infer_nrows: int, skiprows: set[int] | None) -> list[str]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows: set[int] = set()
        buffer_rows: list[str] = []
        detect_rows: list[str] = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer: abc.Iterator[str] = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(self, infer_nrows: int = 100, skiprows: set[int] | None = None) -> list[tuple[int, int]]:
        delimiters: str = ''.join([f'\\{x}' for x in self.delimiter])
        pattern: re.Pattern[str] = re.compile(f'([^{delimiters}]+)')
        rows: list[str] = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError('No rows from which to infer column width')
        max_len: int = max(map(len, rows))
        mask: np.ndarray = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows: list[str] = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start():m.end()] = 1
        shifted: np.ndarray = np.roll(mask, 1)
        shifted[0] = 0
        edges: np.ndarray = np.where(mask ^ shifted == 1)[0]
        edge_pairs: list[tuple[int, int]] = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self) -> list[str]:
        if self.buffer is not None:
            try:
                line: str = next(self.buffer)
            except StopIteration:
                self.buffer: abc.Iterator[str] | None = None
                line: str = next(self.f)
        else:
            line: str = next(self.f)
        return [line[from_:to].strip(self.delimiter) for from_, to in self.colspecs]

class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f: IO[str], **kwds: Any) -> None:
        self.colspecs: list[tuple[int, int]] = kwds.pop('colspecs')
        self.infer_nrows: int = kwds.pop('infer_nrows')
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str]) -> abc.Iterator[str]:
        return FixedWidthReader(f, self.colspecs, self.delimiter, self.comment, self.skiprows, self.infer_nrows)

    def _remove_empty_lines(self, lines: list[str]) -> list[str]:
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """
        return [line for line in lines if any((not isinstance(e, str) or e.strip() for e in line))]

def _validate_skipfooter_arg(skipfooter: int) -> int:
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : non-negative integer
        The number of rows to skip at the end of the file.

    Returns
    -------
    validated_skipfooter : non-negative integer
        The original input if the validation succeeds.

    Raises
    ------
    ValueError : 'skipfooter' was not a non-negative integer.
    """
    if not is_integer(skipfooter):
        raise ValueError('skipfooter must be an integer')
    if skipfooter < 0:
        raise ValueError('skipfooter cannot be negative')
    return skipfooter
