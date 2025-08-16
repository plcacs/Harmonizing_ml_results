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

    def __init__(self, f: IO, **kwds: Any) -> None:
        super().__init__(kwds)
        self.data: list = []
        self.buf: list = []
        self.pos: int = 0
        self.line_pos: int = 0
        self.skiprows: Any = kwds['skiprows']
        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = lambda x: x in self.skiprows
        self.skipfooter: int = _validate_skipfooter_arg(kwds['skipfooter'])
        self.delimiter: Any = kwds['delimiter']
        self.quotechar: Any = kwds['quotechar']
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)
        self.escapechar: Any = kwds['escapechar']
        self.doublequote: Any = kwds['doublequote']
        self.skipinitialspace: Any = kwds['skipinitialspace']
        self.lineterminator: Any = kwds['lineterminator']
        self.quoting: Any = kwds['quoting']
        self.skip_blank_lines: Any = kwds['skip_blank_lines']
        self.has_index_names: bool = kwds.get('has_index_names', False)
        self.thousands: Any = kwds['thousands']
        self.decimal: Any = kwds['decimal']
        self.comment: Any = kwds['comment']
        if isinstance(f, list):
            self.data = f
        else:
            assert hasattr(f, 'readline')
            self.data = self._make_reader(f)
        self._col_indices: Any = None
        columns, self.num_original_columns, self.unnamed_cols = self._infer_columns()
        self.columns, self.index_names, self.col_names, _ = self._extract_multi_indexer_columns(columns, self.index_names)
        self.orig_names: list = list(self.columns)
        index_names, self.orig_names, self.columns = self._get_index_name()
        if self.index_names is None:
            self.index_names = index_names
        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns))
        self._no_thousands_columns: set = self._set_no_thousand_columns()
        if len(self.decimal) != 1:
            raise ValueError('Only length-1 decimal markers supported')

    @cache_readonly
    def num(self) -> re.Pattern:
        decimal: str = re.escape(self.decimal)
        if self.thousands is None:
            regex: str = f'^[\\-\\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\\-?[0-9]+)?$'
        else:
            thousands: str = re.escape(self.thousands)
            regex: str = f'^[\\-\\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?([0-9]?(E|e)\\-?[0-9]+)?$'
        return re.compile(regex)

    def _make_reader(self, f: IO) -> Any:
        sep: Any = self.delimiter
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
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))
            reader = csv.reader(f, dialect=dia, strict=True)
        else:

            def _read():
                line = f.readline()
                pat = re.compile(sep)
                yield pat.split(line.strip())
                for line in f:
                    yield pat.split(line.strip())
            reader = _read()
        return reader

    def read(self, rows: Any = None) -> tuple:
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

    def _exclude_implicit_index(self, alldata: dict) -> tuple:
        names = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
        offset: int = 0
        if self._implicit_index:
            offset = len(self.index_col)
        len_alldata: int = len(alldata)
        self._check_data_length(names, alldata)
        return ({name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata}, names)

    def get_chunk(self, size: Any = None) -> tuple:
        if size is None:
            size = self.chunksize
        return self.read(rows=size)

    def _convert_data(self, data: dict) -> dict:
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
    def _convert_to_ndarrays(self, dct: dict, na_values: dict, na_fvalues: dict, converters: Any = None, dtypes: Any = None) -> dict:
        result: dict = {}
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
    def _cast_types(self, values: Any, cast_type: Any, column: str) -> Any:
        if isinstance(cast_type, CategoricalDtype):
            known_cats: bool = cast_type.categories is not None
            if not is_object_dtype(values.dtype) and (not known_cats):
                values = lib.ensure_string_array(values, skipna=False, convert_na_value=False)
            cats: Index = Index(values).unique().dropna()
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

    def _infer_columns(self) -> tuple:
        names: Any = self.names
        num_original_columns: int = 0
        clear_buffer: bool = True
        unnamed_cols: set = set()
        if self.header is not None:
            header = self.header
            have_mi_columns = self._have_mi_columns
            if isinstance(header, (list, tuple, np.ndarray)):
                if have_mi_columns:
                    header = list(header) + [header[-1] + 1]
            else:
                header = [header]
            columns: list = []
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
    def _header_line(self) -> Any:
        if self.header is None:
            return None
        try:
            line = self._buffered_line()
        except StopIteration as err:
            if not self.names:
                raise EmptyDataError('No columns