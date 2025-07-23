from __future__ import annotations
from collections import abc, defaultdict
import csv
from io import StringIO
import re
from typing import IO, TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union, cast, final
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

    def __init__(self, f: Union[IO[str], List[str]], **kwds: Any) -> None:
        """
        Workhorse function for processing nested list into DataFrame
        """
        super().__init__(kwds)
        self.data: Union[List[str], Iterator[List[str]]] = []
        self.buf: List[List[str]] = []
        self.pos: int = 0
        self.line_pos: int = 0
        self.skiprows = kwds['skiprows']
        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = lambda x: x in self.skiprows
        self.skipfooter: int = _validate_skipfooter_arg(kwds['skipfooter'])
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
        self._col_indices: Optional[List[int]] = None
        columns, self.num_original_columns, self.unnamed_cols = self._infer_columns()
        self.columns, self.index_names, self.col_names, _ = self._extract_multi_indexer_columns(columns, self.index_names)
        self.orig_names = list(self.columns)
        index_names, self.orig_names, self.columns = self._get_index_name()
        if self.index_names is None:
            self.index_names = index_names
        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns)))
        self._no_thousands_columns: Set[int] = self._set_no_thousand_columns()
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

    def _make_reader(self, f: IO[str]) -> Iterator[List[str]]:
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

            def _read() -> Iterator[List[str]]:
                line = f.readline()
                pat = re.compile(sep)
                yield pat.split(line.strip())
                for line in f:
                    yield pat.split(line.strip())
            reader = _read()
        return reader

    def read(self, rows: Optional[int] = None) -> Tuple[Index, List[Hashable], Dict[Hashable, ArrayLike]]:
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
        if self.has_index_names and sum((int(v == '' or v is None) for v in content[0]) == len(columns):
            indexnamerow = content[0]
            content = content[1:]
        alldata = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)
        conv_data = self._convert_data(data)
        conv_data = self._do_date_conversions(columns, conv_data)
        index, result_columns = self._make_index(alldata, columns, indexnamerow)
        return (index, result_columns, conv_data)

    def _exclude_implicit_index(self, alldata: List[ArrayLike]) -> Tuple[Dict[Hashable, ArrayLike], List[Hashable]]:
        names = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
        offset = 0
        if self._implicit_index:
            offset = len(self.index_col)
        len_alldata = len(alldata)
        self._check_data_length(names, alldata)
        return ({name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata}, names)

    def get_chunk(self, size: Optional[int] = None) -> Tuple[Index, List[Hashable], Dict[Hashable, ArrayLike]]:
        if size is None:
            size = self.chunksize
        return self.read(rows=size)

    def _convert_data(self, data: Dict[Hashable, ArrayLike]) -> Dict[Hashable, ArrayLike]:
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
    def _convert_to_ndarrays(self, dct: Dict[Hashable, ArrayLike], na_values: Union[Set[Scalar], Dict[Hashable, Set[Scalar]]], na_fvalues: Union[Set[Scalar], Dict[Hashable, Set[Scalar]]], converters: Optional[Dict[Hashable, Any]] = None, dtypes: Optional[Dict[Hashable, DtypeObj]] = None) -> Dict[Hashable, ArrayLike]:
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
    def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column: Hashable) -> ArrayLike:
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

    def _infer_columns(self) -> Tuple[List[List[str]], int, Set[str]]:
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols: Set[str] = set()
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
                        msg = f'[{','.join(joi)}], len of {len(joi)}, '
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
                    counts: DefaultDict[str, int] = defaultdict(int)
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
                elif