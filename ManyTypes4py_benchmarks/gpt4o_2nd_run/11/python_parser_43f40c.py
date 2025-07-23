from __future__ import annotations
from collections import abc, defaultdict
import csv
from io import StringIO
import re
from typing import IO, TYPE_CHECKING, Any, DefaultDict, Literal, cast, final, Union, Optional
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

_BOM = '\ufeff'

class PythonParser(ParserBase):

    def __init__(self, f: Union[IO[str], list[str]], **kwds: Any) -> None:
        super().__init__(kwds)
        self.data: list[str] = []
        self.buf: list[list[str]] = []
        self.pos: int = 0
        self.line_pos: int = 0
        self.skiprows = kwds['skiprows']
        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = lambda x: x in self.skiprows
        self.skipfooter = _validate_skipfooter_arg(kwds['skipfooter'])
        self.delimiter: Optional[str] = kwds['delimiter']
        self.quotechar: Optional[str] = kwds['quotechar']
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)
        self.escapechar: Optional[str] = kwds['escapechar']
        self.doublequote: bool = kwds['doublequote']
        self.skipinitialspace: bool = kwds['skipinitialspace']
        self.lineterminator: Optional[str] = kwds['lineterminator']
        self.quoting: int = kwds['quoting']
        self.skip_blank_lines: bool = kwds['skip_blank_lines']
        self.has_index_names: bool = kwds.get('has_index_names', False)
        self.thousands: Optional[str] = kwds['thousands']
        self.decimal: str = kwds['decimal']
        self.comment: Optional[str] = kwds['comment']
        if isinstance(f, list):
            self.data = f
        else:
            assert hasattr(f, 'readline')
            self.data = self._make_reader(f)
        self._col_indices: Optional[list[int]] = None
        columns, self.num_original_columns, self.unnamed_cols = self._infer_columns()
        self.columns, self.index_names, self.col_names, _ = self._extract_multi_indexer_columns(columns, self.index_names)
        self.orig_names: list[str] = list(self.columns)
        index_names, self.orig_names, self.columns = self._get_index_name()
        if self.index_names is None:
            self.index_names = index_names
        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns)))
        self._no_thousands_columns = self._set_no_thousand_columns()
        if len(self.decimal) != 1:
            raise ValueError('Only length-1 decimal markers supported')

    @cache_readonly
    def num(self) -> re.Pattern:
        decimal = re.escape(self.decimal)
        if self.thousands is None:
            regex = f'^[\\-\\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\\-?[0-9]+)?$'
        else:
            thousands = re.escape(self.thousands)
            regex = f'^[\\-\\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?([0-9]?(E|e)\\-?[0-9]+)?$'
        return re.compile(regex)

    def _make_reader(self, f: IO[str]) -> Union[csv.reader, Iterator[list[str]]]:
        sep = self.delimiter
        if sep is None or len(sep) == 1:
            if self.lineterminator:
                raise ValueError('Custom line terminators not supported in python parser (yet)')

            class MyDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = '\n'
            dia = MyDialect
            if sep is not None:
                dia.delimiter = sep
            else:
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]
                lines_str = cast(list[str], lines)
                line = lines_str[0]
                self.pos += 1
                self.line_pos += 1
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))
            reader = csv.reader(f, dialect=dia, strict=True)
        else:

            def _read() -> Iterator[list[str]]:
                line = f.readline()
                pat = re.compile(sep)
                yield pat.split(line.strip())
                for line in f:
                    yield pat.split(line.strip())
            reader = _read()
        return reader

    def read(self, rows: Optional[int] = None) -> tuple[Index, list[str], dict[str, np.ndarray]]:
        try:
            content = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content = []
            else:
                self.close()
                raise
        self._first_chunk = False
        columns = list(self.orig_names)
        if not len(content):
            names = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
            index, columns, col_dict = self._get_empty_meta(names, self.dtype)
            conv_columns = self._maybe_make_multi_index_columns(columns, self.col_names)
            return (index, conv_columns, col_dict)
        indexnamerow = None
        if self.has_index_names and sum((int(v == '' or v is None) for v in content[0])) == len(columns):
            indexnamerow = content[0]
            content = content[1:]
        alldata = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)
        conv_data = self._convert_data(data)
        conv_data = self._do_date_conversions(columns, conv_data)
        index, result_columns = self._make_index(alldata, columns, indexnamerow)
        return (index, result_columns, conv_data)

    def _exclude_implicit_index(self, alldata: list[list[Any]]) -> tuple[dict[str, list[Any]], list[str]]:
        names = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
        offset = 0
        if self._implicit_index:
            offset = len(self.index_col)
        len_alldata = len(alldata)
        self._check_data_length(names, alldata)
        return ({name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata}, names)

    def get_chunk(self, size: Optional[int] = None) -> tuple[Index, list[str], dict[str, np.ndarray]]:
        if size is None:
            size = self.chunksize
        return self.read(rows=size)

    def _convert_data(self, data: dict[str, list[Any]]) -> dict[str, np.ndarray]:
        clean_conv = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)
        clean_na_values = {}
        clean_na_fvalues = {}
        if isinstance(self.na_values, dict):
            for col in self.na_values:
                if col is not None:
                    na_value = self.na_values[col]
                    na_fvalue = self.na_fvalues[col]
                    if isinstance(col, int) and col not in self.orig_names:
                        col = self.orig_names[col]
                    clean_na_values[col] = na_value
                    clean_na_fvalues[col] = na_fvalue
        else:
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues
        return self._convert_to_ndarrays(data, clean_na_values, clean_na_fvalues, clean_conv, clean_dtypes)

    @final
    def _convert_to_ndarrays(self, dct: dict[str, list[Any]], na_values: dict[str, set[Any]], na_fvalues: dict[str, set[Any]], converters: Optional[dict[str, Any]] = None, dtypes: Optional[dict[str, Any]] = None) -> dict[str, np.ndarray]:
        result = {}
        parse_date_cols = validate_parse_dates_presence(self.parse_dates, self.columns)
        for c, values in dct.items():
            conv_f = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                cast_type = dtypes
            if self.na_filter:
                col_na_values, col_na_fvalues = get_na_values(c, na_values, na_fvalues, self.keep_default_na)
            else:
                col_na_values, col_na_fvalues = (set(), set())
            if c in parse_date_cols:
                mask = algorithms.isin(values, set(col_na_values) | col_na_fvalues)
                np.putmask(values, mask, np.nan)
                result[c] = values
                continue
            if conv_f is not None:
                if cast_type is not None:
                    warnings.warn(f'Both a converter and dtype were specified for column {c} - only the converter will be used.', ParserWarning, stacklevel=find_stack_level())
                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)
                cvals, na_count = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool=False)
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
                try_num_bool = not (cast_type and is_str_or_ea_dtype)
                cvals, na_count = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool)
                if cast_type is not None:
                    cast_type = pandas_dtype(cast_type)
                if cast_type and (cvals.dtype != cast_type or is_ea):
                    if not is_ea and na_count > 0:
                        if is_bool_dtype(cast_type):
                            raise ValueError(f'Bool column has NA values in column {c}')
                    cvals = self._cast_types(cvals, cast_type, c)
            result[c] = cvals
        return result

    @final
    def _cast_types(self, values: Union[np.ndarray, ExtensionArray], cast_type: Union[np.dtype, ExtensionDtype], column: str) -> Union[np.ndarray, ExtensionArray]:
        if isinstance(cast_type, CategoricalDtype):
            known_cats = cast_type.categories is not None
            if not is_object_dtype(values.dtype) and (not known_cats):
                values = lib.ensure_string_array(values, skipna=False, convert_na_value=False)
            cats = Index(values).unique().dropna()
            values = Categorical._from_inferred_categories(cats, cats.get_indexer(values), cast_type, true_values=self.true_values)
        elif isinstance(cast_type, ExtensionDtype):
            array_type = cast_type.construct_array_type()
            try:
                if isinstance(cast_type, BooleanDtype):
                    values_str = [str(val) for val in values]
                    return array_type._from_sequence_of_strings(values_str, dtype=cast_type, true_values=self.true_values, false_values=self.false_values, none_values=self.na_values)
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(f'Extension Array: {array_type} must implement _from_sequence_of_strings in order to be used in parser methods') from err
        elif isinstance(values, ExtensionArray):
            values = values.astype(cast_type, copy=False)
        elif issubclass(cast_type.type, str):
            values = lib.ensure_string_array(values, skipna=True, convert_na_value=False)
        else:
            try:
                values = astype_array(values, cast_type, copy=True)
            except ValueError as err:
                raise ValueError(f'Unable to convert column {column} to type {cast_type}') from err
        return values

    @cache_readonly
    def _have_mi_columns(self) -> bool:
        if self.header is None:
            return False
        header = self.header
        if isinstance(header, (list, tuple, np.ndarray)):
            return len(header) > 1
        else:
            return False

    def _infer_columns(self) -> tuple[list[list[str]], int, set[str]]:
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols = set()
        if self.header is not None:
            header = self.header
            have_mi_columns = self._have_mi_columns
            if isinstance(header, (list, tuple, np.ndarray)):
                if have_mi_columns:
                    header = list(header) + [header[-1] + 1]
            else:
                header = [header]
            columns = []
            for level, hr in enumerate(header):
                try:
                    line = self._buffered_line()
                    while self.line_pos <= hr:
                        line = self._next_line()
                except StopIteration as err:
                    if 0 < self.line_pos <= hr and (not have_mi_columns or hr != header[-1]):
                        joi = list(map(str, header[:-1] if have_mi_columns else header))
                        msg = f'[{",".join(joi)}], len of {len(joi)}, '
                        raise ValueError(f'Passed header={msg}but only {self.line_pos} lines in file') from err
                    if have_mi_columns and hr > 0:
                        if clear_buffer:
                            self.buf.clear()
                        columns.append([None] * len(columns[-1]))
                        return (columns, num_original_columns, unnamed_cols)
                    if not self.names:
                        raise EmptyDataError('No columns to parse from file') from err
                    line = self.names[:]
                this_columns = []
                this_unnamed_cols = []
                for i, c in enumerate(line):
                    if c == '':
                        if have_mi_columns:
                            col_name = f'Unnamed: {i}_level_{level}'
                        else:
                            col_name = f'Unnamed: {i}'
                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)
                if not have_mi_columns:
                    counts = defaultdict(int)
                    col_loop_order = [i for i in range(len(this_columns)) if i not in this_unnamed_cols] + this_unnamed_cols
                    for i in col_loop_order:
                        col = this_columns[i]
                        old_col = col
                        cur_count = counts[col]
                        if cur_count > 0:
                            while cur_count > 0:
                                counts[old_col] = cur_count + 1
                                col = f'{old_col}.{cur_count}'
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
                        lc = len(this_columns)
                        sic = self.index_col
                        ic = len(sic) if sic is not None else 0
                        unnamed_count = len(this_unnamed_cols)
                        if lc != unnamed_count and lc - ic > unnamed_count or ic == 0:
                            clear_buffer = False
                            this_columns = [None] * lc
                            self.buf = [self.buf[-1]]
                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})
                if len(columns) == 1:
                    num_original_columns = len(this_columns)
            if clear_buffer:
                self.buf.clear()
            if names is not None:
                try:
                    first_line = self._next_line()
                except StopIteration:
                    first_line = None
                len_first_data_row = 0 if first_line is None else len(first_line)
                if len(names) > len(columns[0]) and len(names) > len_first_data_row:
                    raise ValueError('Number of passed names did not match number of header fields in the file')
                if len(columns) > 1:
                    raise TypeError('Cannot pass names with multi-index columns')
                if self.usecols is not None:
                    self._handle_usecols(columns, names, num_original_columns)
                else:
                    num_original_columns = len(names)
                if self._col_indices is not None and len(names) != len(self._col_indices):
                    columns = [[names[i] for i in sorted(self._col_indices)]]
                else:
                    columns = [names]
            else:
                columns = self._handle_usecols(columns, columns[0], num_original_columns)
        else:
            ncols = len(self._header_line)
            num_original_columns = ncols
            if not names:
                columns = [list(range(ncols))]
                columns = self._handle_usecols(columns, columns[0], ncols)
            elif self.usecols is None or len(names) >= ncols:
                columns = self._handle_usecols([names], names, ncols)
                num_original_columns = len(names)
            elif not callable(self.usecols) and len(names) != len(self.usecols):
                raise ValueError('Number of passed names did not match number of header fields in the file')
            else:
                columns = [names]
                self._handle_usecols(columns, columns[0], ncols)
        return (columns, num_original_columns, unnamed_cols)

    @cache_readonly
    def _header_line(self) -> Optional[list[str]]:
        if self.header is not None:
            return None
        try:
            line = self._buffered_line()
        except StopIteration as err:
            if not self.names:
                raise EmptyDataError('No columns to parse from file') from err
            line = self.names[:]
        return line

    def _handle_usecols(self, columns: list[list[str]], usecols_key: list[str], num_original_columns: int) -> list[list[str]]:
        if self.usecols is not None:
            if callable(self.usecols):
                col_indices = evaluate_callable_usecols(self.usecols, usecols_key)
            elif any((isinstance(u, str) for u in self.usecols)):
                if len(columns) > 1:
                    raise ValueError('If using multiple headers, usecols must be integers.')
                col_indices = []
                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            self._validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                missing_usecols = [col for col in self.usecols if col >= num_original_columns]
                if missing_usecols:
                    raise ParserError(f'Defining usecols with out-of-bounds indices is not allowed. {missing_usecols} are out-of-bounds.')
                col_indices = self.usecols
            columns = [[n for i, n in enumerate(column) if i in col_indices] for column in columns]
            self._col_indices = sorted(col_indices)
        return columns

    def _buffered_line(self) -> list[str]:
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row: list[str]) -> list[str]:
        if not first_row:
            return first_row
        if not isinstance(first_row[0], str):
            return first_row
        if not first_row[0]:
            return first_row
        first_elt = first_row[0][0]
        if first_elt != _BOM:
            return first_row
        first_row_bom = first_row[0]
        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start = 2
            quote = first_row_bom[1]
            end = first_row_bom[2:].index(quote) + 2
            new_row = first_row_bom[start:end]
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1:]
        else:
            new_row = first_row_bom[1:]
        new_row_list = [new_row]
        return new_row_list + first_row[1:]

    def _is_line_empty(self, line: Union[str, list[Any]]) -> bool:
        return not line or all((not x for x in line))

    def _next_line(self) -> list[str]:
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                if self.pos >= len(self.data):
                    break
                self.pos += 1
            while True:
                try:
                    line = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    if not self.skip_blank_lines and (self._is_line_empty(self.data[self.pos - 1]) or line):
                        break
                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError as err:
                    raise StopIteration from err
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                next(self.data)
            while True:
                orig_line = self._next_iter_line(row_num=self.pos + 1)
                self.pos += 1
                if orig_line is not None:
                    line = self._check_comments([orig_line])[0]
                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                    elif self._is_line_empty(orig_line) or line:
                        break
        if self.pos == 1:
            line = self._check_for_bom(line)
        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg: str, row_num: int) -> None:
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        if self.on_bad_lines == self.BadLineHandleMethod.WARN:
            warnings.warn(f'Skipping line {row_num}: {msg}\n', ParserWarning, stacklevel=find_stack_level())

    def _next_iter_line(self, row_num: int) -> Optional[str]:
        try:
            assert not isinstance(self.data, list)
            line = next(self.data)
            return line
        except csv.Error as e:
            if self.on_bad_lines in (self.BadLineHandleMethod.ERROR, self.BadLineHandleMethod.WARN):
                msg = str(e)
                if 'NULL byte' in msg or 'line contains NUL' in msg:
                    msg = "NULL byte detected. This byte cannot be processed in Python's native csv library at the moment, so please pass in engine='c' instead"
                if self.skipfooter > 0:
                    reason = "Error could possibly be due to parsing errors in the skipped footer rows (the skipfooter keyword is only applied after Python's csv library has parsed all rows)."
                    msg += '. ' + reason
                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines: list[list[str]]) -> list[list[str]]:
        if self.comment is None:
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if not isinstance(x, str) or self.comment not in x or x in self.na_values:
                    rl.append(x)
                else:
                    x = x[:x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines: list[list[Any]]) -> list[list[Any]]:
        ret = [line for line in lines if len(line) > 1 or (len(line) == 1 and (not isinstance(line[0], str) or line[0].strip()))]
        return ret

    def _check_thousands(self, lines: list[list[str]]) -> list[list[str]]:
        if self.thousands is None:
            return lines
        return self._search_replace_num_columns(lines=lines, search=self.thousands, replace='')

    def _search_replace_num_columns(self, lines: list[list[str]], search: str, replace: str) -> list[list[str]]:
        ret = []
        for line in lines:
            rl = []
            for i, x in enumerate(line):
                if not isinstance(x, str) or search not in x or i in self._no_thousands_columns or (not self.num.search(x.strip())):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines: list[list[str]]) -> list[list[str]]:
        if self.decimal == parser_defaults['decimal']:
            return lines
        return self._search_replace_num_columns(lines=lines, search=self.decimal, replace='.')

    def _get_index_name(self) -> tuple[Optional[list[str]], list[str], list[str]]:
        columns = self.orig_names
        orig_names = list(columns)
        columns = list(columns)
        if self._header_line is not None:
            line = self._header_line
        else:
            try:
                line = self._next_line()
            except StopIteration:
                line = None
        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None
        implicit_first_cols = 0
        if line is not None:
            index_col = self.index_col
            if index_col is not False:
                implicit_first_cols = len(line) - self.num_original_columns
            if next_line is not None and self.header is not None and (index_col is not False):
                if len(next_line) == len(line) + self.num_original_columns:
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]
                    for c in reversed(line):
                        columns.insert(0, c)
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return (line, orig_names, columns)
        if implicit_first_cols > 0:
            self._implicit_index = True
            if self.index_col is None:
                self.index_col = list(range(implicit_first_cols))
            index_name = None
        else:
            index_name, _, self.index_col = self._clean_index_names(columns, self.index_col)
        return (index_name, orig_names, columns)

    def _rows_to_cols(self, content: list[list[Any]]) -> list[list[Any]]:
        col_len = self.num_original_columns
        if self._implicit_index:
            col_len += len(self.index_col)
        max_len = max((len(row) for row in content))
        if max_len > col_len and self.index_col is not False and (self.usecols is None):
            footers = self.skipfooter if self.skipfooter else 0
            bad_lines = []
            iter_content = enumerate(content)
            content_len = len(content)
            content = []
            for i, _content in iter_content:
                actual_len = len(_content)
                if actual_len > col_len:
                    if callable(self.on_bad_lines):
                        new_l = self.on_bad_lines(_content)
                        if new_l is not None:
                            content.append(new_l)
                    elif self.on_bad_lines in (self.BadLineHandleMethod.ERROR, self.BadLineHandleMethod.WARN):
                        row_num = self.pos - (content_len - i + footers)
                        bad_lines.append((row_num, actual_len))
                        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                            break
                else:
                    content.append(_content)
            for row_num, actual_len in bad_lines:
                msg = f'Expected {col_len} fields in line {row_num + 1}, saw {actual_len}'
                if self.delimiter and len(self.delimiter) > 1 and (self.quoting != csv.QUOTE_NONE):
                    reason = 'Error could possibly be due to quotes being ignored when a multi-char delimiter is used.'
                    msg += '. ' + reason
                self._alert_malformed(msg, row_num + 1)
        zipped_content = list(lib.to_object_array(content, min_width=col_len).T)
        if self.usecols:
            assert self._col_indices is not None
            col_indices = self._col_indices
            if self._implicit_index:
                zipped_content = [a for i, a in enumerate(zipped_content) if i < len(self.index_col) or i - len(self.index_col) in col_indices]
            else:
                zipped_content = [a for i, a in enumerate(zipped_content) if i in col_indices]
        return zipped_content

    def _get_lines(self, rows: Optional[int] = None) -> list[list[str]]:
        lines = self.buf
        new_rows = None
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
                    new_rows = self.data[self.pos:]
                    new_pos = len(self.data)
                else:
                    new_rows = self.data[self.pos:self.pos + rows]
                    new_pos = self.pos + rows
                new_rows = self._remove_skipped_rows(new_rows)
                lines.extend(new_rows)
                self.pos = new_pos
            else:
                new_rows = []
                try:
                    if rows is not None:
                        row_index = 0
                        row_ct = 0
                        offset = self.pos if self.pos is not None else 0
                        while row_ct < rows:
                            new_row = next(self.data)
                            if not self.skipfunc(offset + row_index):
                                row_ct += 1
                            row_index += 1
                            new_rows.append(new_row)
                        len_new_rows = len(new_rows)
                        new_rows = self._remove_skipped_rows(new_rows)
                        lines.extend(new_rows)
                    else:
                        rows = 0
                        while True:
                            next_row = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1
                            if next_row is not None:
                                new_rows.append(next_row)
                        len_new_rows = len(new_rows)
                except StopIteration:
                    len_new_rows = len(new_rows)
                    new_rows = self._remove_skipped_rows(new_rows)
                    lines.extend(new_rows)
                    if len(lines) == 0:
                        raise
                self.pos += len_new_rows
            self.buf = []
        else:
            lines = new_rows
        if self.skipfooter:
            lines = lines[:-self.skipfooter]
        lines = self._check_comments(lines)
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        lines = self._check_thousands(lines)
        return self._check_decimal(lines)

    def _remove_skipped_rows(self, new_rows: list[str]) -> list[str]:
        if self.skiprows:
            return [row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)]
        return new_rows

    def _set_no_thousand_columns(self) -> set[int]:
        no_thousands_columns = set()
        if self.columns and self.parse_dates:
            assert self._col_indices is not None
            no_thousands_columns = self._set_noconvert_dtype_columns(self._col_indices, self.columns)
        if self.columns and self.dtype:
            assert self._col_indices is not None
            for i, col in zip(self._col_indices, self.columns):
                if not isinstance(self.dtype, dict) and (not is_numeric_dtype(self.dtype)):
                    no_thousands_columns.add(i)
                if isinstance(self.dtype, dict) and col in self.dtype and (not is_numeric_dtype(self.dtype[col]) or is_bool_dtype(self.dtype[col])):
                    no_thousands_columns.add(i)
        return no_thousands_columns

class FixedWidthReader(abc.Iterator):
    def __init__(self, f: IO[str], colspecs: Union[str, list[tuple[Optional[int], Optional[int]]]], delimiter: Optional[str], comment: Optional[str], skiprows: Optional[set[int]] = None, infer_nrows: int = 100) -> None:
        self.f = f
        self.buffer: Optional[Iterator[str]] = None
        self.delimiter = '\r\n' + delimiter if delimiter else '\n\r\t '
        self.comment = comment
        if colspecs == 'infer':
            self.colspecs = self.detect_colspecs(infer_nrows=infer_nrows, skiprows=skiprows)
        else:
            self.colspecs = colspecs
        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(f'column specifications must be a list or tuple, input was a {type(colspecs).__name__}')
        for colspec in self.colspecs:
            if not (isinstance(colspec, (tuple, list)) and len(colspec) == 2 and isinstance(colspec[0], (int, np.integer, type(None))) and isinstance(colspec[1], (int, np.integer, type(None)))):
                raise TypeError('Each column specification must be 2 element tuple or list of integers')

    def get_rows(self, infer_nrows: int, skiprows: Optional[set[int]] = None) -> list[str]:
        if skiprows is None:
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(self, infer_nrows: int = 100, skiprows: Optional[set[int]] = None) -> list[tuple[int, int]]:
        delimiters = ''.join([f'\\{x}' for x in self.delimiter])
        pattern = re.compile(f'([^{delimiters}]+)')
        rows = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError('No rows from which to infer column width')
        max_len = max(map(len, rows))
        mask = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start():m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where(mask ^ shifted == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self) -> list[str]:
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)
        else:
            line = next(self.f)
        return [line[from_:to].strip(self.delimiter) for from_, to in self.colspecs]

class FixedWidthFieldParser(PythonParser):
    def __init__(self, f: IO[str], **kwds: Any) -> None:
        self.colspecs = kwds.pop('colspecs')
        self.infer_nrows = kwds.pop('infer_nrows')
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str]) -> FixedWidthReader:
        return FixedWidthReader(f, self.colspecs, self.delimiter, self.comment, self.skiprows, self.infer_nrows)

    def _remove_empty_lines(self, lines: list[list[Any]]) -> list[list[Any]]:
        return [line for line in lines if any((not isinstance(e, str) or e.strip() for e in line))]

def _validate_skipfooter_arg(skipfooter: int) -> int:
    if not is_integer(skipfooter):
        raise ValueError('skipfooter must be an integer')
    if skipfooter < 0:
        raise ValueError('skipfooter cannot be negative')
    return skipfooter
