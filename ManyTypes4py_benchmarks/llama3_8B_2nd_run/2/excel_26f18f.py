class ExcelFormatter:
    """
    Class for formatting a DataFrame to a list of ExcelCells,
    Parameters
    ----------
    df : DataFrame or Styler
    na_rep: na representation
    float_format : str, default None
        Format string for floating point numbers
    cols : sequence, optional
        Columns to write
    header : bool or sequence of str, default True
        Write out column names. If a list of string is given it is
        assumed to be aliases for the column names
    index : bool, default True
        output row names (index)
    index_label : str or sequence, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.
    merge_cells : bool or 'columns', default False
        Format MultiIndex column headers and Hierarchical Rows as merged cells
        if True. Merge MultiIndex column headers only if 'columns'.
        .. versionchanged:: 3.0.0
            Added the 'columns' option.
    inf_rep : str, default `'inf'`
        representation for np.inf values (which aren't representable in Excel)
        A `'-'` sign will be added in front of -inf.
    style_converter : callable, optional
        This translates Styler styles (CSS) into ExcelWriter styles.
        Defaults to ``CSSToExcelConverter()``.
        It should have signature css_declarations string -> excel style.
        This is only called for body cells.
    """
    max_rows: int = 2 ** 20
    max_cols: int = 2 ** 14

    def __init__(self, df: DataFrame | Styler, na_rep: str = '', float_format: str | None = None, cols: Sequence[str] | None = None, header: bool | Sequence[str] | None = True, index: bool = True, index_label: str | Sequence[str] | None = None, merge_cells: bool | str = False, inf_rep: str = 'inf', style_converter: Callable[[str], dict] | None = None) -> None:
        self.rowcounter: int = 0
        self.na_rep: str = na_rep
        if not isinstance(df, DataFrame):
            self.styler: Styler | None = df
            self.styler._compute()
            df: DataFrame = df.data
            if style_converter is None:
                style_converter: Callable[[str], dict] = CSSToExcelConverter()
            self.style_converter: Callable[[str], dict] = style_converter
        else:
            self.styler: Styler | None = None
            self.style_converter: Callable[[str], dict] | None = None
        self.df: DataFrame = df
        if cols is not None:
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError('passes columns are not ALL present dataframe')
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df: DataFrame = df.reindex(columns=cols)
        self.columns: Index = self.df.columns
        self.float_format: str | None = float_format
        self.index: bool = index
        self.index_label: str | Sequence[str] | None = index_label
        self.header: bool | Sequence[str] | None = header
        if not isinstance(merge_cells, bool) and merge_cells != 'columns':
            raise ValueError(f'Unexpected value for merge_cells={merge_cells!r}.')
        self.merge_cells: bool | str = merge_cells
        self.inf_rep: str = inf_rep

    def _format_value(self, val: Any) -> str:
        if is_scalar(val) and missing.isna(val):
            val: str = self.na_rep
        elif is_float(val):
            if missing.isposinf_scalar(val):
                val: str = self.inf_rep
            elif missing.isneginf_scalar(val):
                val: str = f'-{self.inf_rep}'
            elif self.float_format is not None:
                val: float = float(self.float_format % val)
        if getattr(val, 'tzinfo', None) is not None:
            raise ValueError('Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.')
        return str(val)

    def _format_header_mi(self) -> Iterator[ExcelCell]:
        if isinstance(self.columns, MultiIndex):
            if not self.index:
                raise NotImplementedError("Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.")
        if not (self._has_aliases or self.header):
            return
        columns: MultiIndex = self.columns
        merge_columns: bool = self.merge_cells in {True, 'columns'}
        level_strs: list[str] = columns._format_multi(sparsify=merge_columns, include_names=False)
        level_lengths: list[tuple[int, ...]] = get_level_lengths(level_strs)
        coloffset: int = 0
        lnum: int = 0
        if self.index and isinstance(self.df.index, MultiIndex):
            coloffset: int = self.df.index.nlevels - 1
        for lnum, name in enumerate(columns.names):
            yield ExcelCell(row=lnum, col=coloffset, val=name, style=None)
        for lnum, (spans, levels, level_codes) in enumerate(zip(level_lengths, columns.levels, columns.codes)):
            values: np.ndarray = levels.take(level_codes)
            for i, span_val in spans.items():
                mergestart: int | None = None
                mergeend: int | None = None
                if merge_columns and span_val > 1:
                    mergestart: int = lnum
                    mergeend: int = coloffset + i + span_val
                yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=values[i], style=None, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
        self.rowcounter: int = lnum

    def _format_header_regular(self) -> Iterator[ExcelCell]:
        if self._has_aliases or self.header:
            coloffset: int = 0
            if self.index:
                coloffset: int = 1
                if isinstance(self.df.index, MultiIndex):
                    coloffset: int = len(self.df.index.names)
            colnames: Index = self.columns
            if self._has_aliases:
                self.header: Sequence[str] = self.header
                if len(self.header) != len(self.columns):
                    raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
                colnames: Index = self.header
            for colindex, colname in enumerate(colnames):
                yield CssExcelCell(row=self.rowcounter, col=colindex + coloffset, val=colname, style=None, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=0, css_col=colindex, css_converter=self.style_converter)

    def _format_header(self) -> Iterator[ExcelCell]:
        if isinstance(self.columns, MultiIndex):
            gen: Iterator[ExcelCell] = self._format_header_mi()
        else:
            gen: Iterator[ExcelCell] = self._format_header_regular()
        gen2: tuple[ExcelCell, ...] = ()
        if self.df.index.names:
            row: list[str] = [x if x is not None else '' for x in self.df.index.names] + [''] * len(self.columns)
            if functools.reduce(lambda x, y: x and y, (x != '' for x in row)):
                gen2: tuple[ExcelCell, ...] = (ExcelCell(self.rowcounter, colindex, val, None) for colindex, val in enumerate(row))
                self.rowcounter: int += 1
        return itertools.chain(gen, gen2)

    def _format_body(self) -> Iterator[ExcelCell]:
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()
        else:
            return self._format_regular_rows()

    def _format_regular_rows(self) -> Iterator[ExcelCell]:
        if self._has_aliases or self.header:
            self.rowcounter: int += 1
        if self.index:
            if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
                index_label: str = self.index_label[0]
            elif self.index_label and isinstance(self.index_label, str):
                index_label: str = self.index_label
            else:
                index_label: str = self.df.index.names[0]
            if isinstance(self.columns, MultiIndex):
                self.rowcounter: int += 1
            if index_label and self.header is not False:
                yield ExcelCell(self.rowcounter - 1, 0, index_label, None)
            index_values: Index = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values: Index = self.df.index.to_timestamp()
            for idx, indexcolval in enumerate(index_values):
                yield CssExcelCell(row=self.rowcounter + idx, col=0, val=indexcolval, style=None, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=0, css_converter=self.style_converter)
            coloffset: int = 1
        else:
            coloffset: int = 0
        yield from self._generate_body(coloffset)

    @property
    def _has_aliases(self) -> bool:
        """Whether the aliases for column names are present."""
        return is_list_like(self.header)

    def _generate_body(self, coloffset: int) -> Iterator[ExcelCell]:
        for colidx in range(len(self.columns)):
            series: Series = self.df.iloc[:, colidx]
            for i, val in enumerate(series):
                yield CssExcelCell(row=self.rowcounter + i, col=colidx + coloffset, val=val, style=None, css_styles=getattr(self.styler, 'ctx', None), css_row=i, css_col=colidx, css_converter=self.style_converter)

    def get_formatted_cells(self) -> Iterator[ExcelCell]:
        for cell in itertools.chain(self._format_header(), self._format_body()):
            cell.val: str = self._format_value(cell.val)
            yield cell

    @doc(storage_options=_shared_docs['storage_options'])
    def write(self, writer: str | file | ExcelWriter, sheet_name: str = 'Sheet1', startrow: int = 0, startcol: int = 0, freeze_panes: tuple[int, int] | None = None, engine: str | None = None, storage_options: dict[str, str] | None = None, engine_kwargs: dict[str, str] | None = None) -> None:
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``,
            or ``io.excel.xlsm.writer``.

        {storage_options}

        engine_kwargs: dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        from pandas.io.excel import ExcelWriter
        num_rows: int = self.df.shape[0]
        num_cols: int = self.df.shape[1]
        if num_rows > self.max_rows or num_cols > self.max_cols:
            raise ValueError(f'This sheet is too large! Your sheet size is: {num_rows}, {num_cols} Max sheet size is: {self.max_rows}, {self.max_cols}')
       