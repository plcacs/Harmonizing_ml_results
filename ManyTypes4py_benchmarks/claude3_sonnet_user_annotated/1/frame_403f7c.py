def _to_internal_pandas(self) -> pd.DataFrame:
    """
    Return a pandas DataFrame directly from _internal to avoid overhead of copy.

    This method is for internal use only.
    """
    return self._internal.to_pandas_frame

def _get_or_create_repr_pandas_cache(self, n: int) -> pd.DataFrame:
    if not hasattr(self, "_repr_pandas_cache") or n not in self._repr_pandas_cache:
        object.__setattr__(
            self, "_repr_pandas_cache", {n: self.head(n + 1)._to_internal_pandas()}
        )
    return self._repr_pandas_cache[n]

def __repr__(self) -> str:
    max_display_count = get_option("display.max_rows")
    if max_display_count is None:
        return self._to_internal_pandas().to_string()

    pdf = self._get_or_create_repr_pandas_cache(max_display_count)
    pdf_length = len(pdf)
    pdf = pdf.iloc[:max_display_count]
    if pdf_length > max_display_count:
        repr_string = pdf.to_string(show_dimensions=True)
        match = REPR_PATTERN.search(repr_string)
        if match is not None:
            nrows = match.group("rows")
            ncols = match.group("columns")
            footer = "\n\n[Showing only the first {nrows} rows x {ncols} columns]".format(
                nrows=nrows, ncols=ncols
            )
            return REPR_PATTERN.sub(footer, repr_string)
    return pdf.to_string()

def _repr_html_(self) -> str:
    max_display_count = get_option("display.max_rows")
    # pandas 0.25.1 has a regression about HTML representation so 'bold_rows'
    # has to be set as False explicitly. See https://github.com/pandas-dev/pandas/issues/28204
    bold_rows = not (LooseVersion("0.25.1") == LooseVersion(pd.__version__))
    if max_display_count is None:
        return self._to_internal_pandas().to_html(notebook=True, bold_rows=bold_rows)

    pdf = self._get_or_create_repr_pandas_cache(max_display_count)
    pdf_length = len(pdf)
    pdf = pdf.iloc[:max_display_count]
    if pdf_length > max_display_count:
        repr_html = pdf.to_html(show_dimensions=True, notebook=True, bold_rows=bold_rows)
        match = REPR_HTML_PATTERN.search(repr_html)
        if match is not None:
            nrows = match.group("rows")
            ncols = match.group("columns")
            by = chr(215)
            footer = (
                "\n<p>Showing only the first {rows} rows "
                "{by} {cols} columns</p>\n</div>".format(rows=nrows, by=by, cols=ncols)
            )
            return REPR_HTML_PATTERN.sub(footer, repr_html)
    return pdf.to_html(notebook=True, bold_rows=bold_rows)

def __getitem__(self, key: Any) -> Union["DataFrame", "Series"]:
    from databricks.koalas.series import Series

    if key is None:
        raise KeyError("none key")
    elif isinstance(key, Series):
        return self.loc[key.astype(bool)]
    elif isinstance(key, slice):
        if any(type(n) == int or None for n in [key.start, key.stop]):
            # Seems like pandas Frame always uses int as positional search when slicing
            # with ints.
            return self.iloc[key]
        return self.loc[key]
    elif is_name_like_value(key):
        return self.loc[:, key]
    elif is_list_like(key):
        return self.loc[:, list(key)]
    raise NotImplementedError(key)

def __setitem__(self, key: Any, value: Any) -> None:
    from databricks.koalas.series import Series

    if isinstance(value, (DataFrame, Series)) and not same_anchor(value, self):
        # Different Series or DataFrames
        level = self._internal.column_labels_level
        key = DataFrame._index_normalized_label(level, key)
        value = DataFrame._index_normalized_frame(level, value)

        def assign_columns(kdf: DataFrame, this_column_labels: List[Tuple], that_column_labels: List[Tuple]) -> Iterator[Tuple[Any, Tuple]]:
            assert len(key) == len(that_column_labels)
            # Note that here intentionally uses `zip_longest` that combine
            # that_columns.
            for k, this_label, that_label in zip_longest(
                key, this_column_labels, that_column_labels
            ):
                yield (kdf._kser_for(that_label), tuple(["that", *k]))
                if this_label is not None and this_label[1:] != k:
                    yield (kdf._kser_for(this_label), this_label)

        kdf = align_diff_frames(assign_columns, self, value, fillna=False, how="left")
    elif isinstance(value, list):
        if len(self) != len(value):
            raise ValueError("Length of values does not match length of index")

        # TODO: avoid using default index?
        with option_context(
            "compute.default_index_type",
            "distributed-sequence",
            "compute.ops_on_diff_frames",
            True,
        ):
            kdf = self.reset_index()
            kdf[key] = ks.DataFrame(value)
            kdf = kdf.set_index(kdf.columns[: self._internal.index_level])
            kdf.index.names = self.index.names

    elif isinstance(key, list):
        assert isinstance(value, DataFrame)
        # Same DataFrames.
        field_names = value.columns
        kdf = self._assign({k: value[c] for k, c in zip(key, field_names)})
    else:
        # Same Series.
        kdf = self._assign({key: value})

    self._update_internal_frame(kdf._internal)

@staticmethod
def _index_normalized_label(level: int, labels: Any) -> List[Tuple]:
    """
    Returns a label that is normalized against the current column index level.
    For example, the key "abc" can be ("abc", "", "") if the current Frame has
    a multi-index for its column
    """
    if is_name_like_tuple(labels):
        labels = [labels]
    elif is_name_like_value(labels):
        labels = [(labels,)]
    else:
        labels = [k if is_name_like_tuple(k) else (k,) for k in labels]

    if any(len(label) > level for label in labels):
        raise KeyError(
            "Key length ({}) exceeds index depth ({})".format(
                max(len(label) for label in labels), level
            )
        )
    return [tuple(list(label) + ([""] * (level - len(label)))) for label in labels]

@staticmethod
def _index_normalized_frame(level: int, kser_or_kdf: Union["Series", "DataFrame"]) -> "DataFrame":
    """
    Returns a frame that is normalized against the current column index level.
    For example, the name in `pd.Series([...], name="abc")` can be can be
    ("abc", "", "") if the current DataFrame has a multi-index for its column
    """
    from databricks.koalas.series import Series

    if isinstance(kser_or_kdf, Series):
        kdf = kser_or_kdf.to_frame()
    else:
        assert isinstance(kser_or_kdf, DataFrame), type(kser_or_kdf)
        kdf = kser_or_kdf.copy()

    kdf.columns = pd.MultiIndex.from_tuples(
        [
            tuple([name_like_string(label)] + ([""] * (level - 1)))
            for label in kdf._internal.column_labels
        ],
    )

    return kdf

def __getattr__(self, key: str) -> Any:
    if key.startswith("__"):
        raise AttributeError(key)
    if hasattr(_MissingPandasLikeDataFrame, key):
        property_or_func = getattr(_MissingPandasLikeDataFrame, key)
        if isinstance(property_or_func, property):
            return property_or_func.fget(self)  # type: ignore
        else:
            return partial(property_or_func, self)

    try:
        return self.loc[:, key]
    except KeyError:
        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__.__name__, key)
        )

def __setattr__(self, key: str, value: Any) -> None:
    try:
        object.__getattribute__(self, key)
        return object.__setattr__(self, key, value)
    except AttributeError:
        pass

    if (key,) in self._internal.column_labels:
        self[key] = value
    else:
        msg = "Koalas doesn't allow columns to be created via a new attribute name"
        if is_testing():
            raise AssertionError(msg)
        else:
            warnings.warn(msg, UserWarning)

def __len__(self) -> int:
    return self._internal.resolved_copy.spark_frame.count()

def __dir__(self) -> List[str]:
    fields = [
        f for f in self._internal.resolved_copy.spark_frame.schema.fieldNames() if " " not in f
    ]
    return super().__dir__() + fields

def __iter__(self) -> Iterator:
    return iter(self.columns)

# NDArray Compat
def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> "DataFrame":
    # TODO: is it possible to deduplicate it with '_map_series_op'?
    if all(isinstance(inp, DataFrame) for inp in inputs) and any(
        not same_anchor(inp, inputs[0]) for inp in inputs
    ):
        # binary only
        assert len(inputs) == 2
        this = inputs[0]
        that = inputs[1]
        if this._internal.column_labels_level != that._internal.column_labels_level:
            raise ValueError("cannot join with no overlapping index names")

        # Different DataFrames
        def apply_op(kdf: DataFrame, this_column_labels: List[Tuple], that_column_labels: List[Tuple]) -> Iterator[Tuple[Any, Tuple]]:
            for this_label, that_label in zip(this_column_labels, that_column_labels):
                yield (
                    ufunc(
                        kdf._kser_for(this_label), kdf._kser_for(that_label), **kwargs
                    ).rename(this_label),
                    this_label,
                )

        return align_diff_frames(apply_op, this, that, fillna=True, how="full")
    else:
        # DataFrame and Series
        applied = []
        this = inputs[0]
        assert all(inp is this for inp in inputs if isinstance(inp, DataFrame))

        for label in this._internal.column_labels:
            arguments = []
            for inp in inputs:
                arguments.append(inp[label] if isinstance(inp, DataFrame) else inp)
            # both binary and unary.
            applied.append(ufunc(*arguments, **kwargs).rename(label))

        internal = this._internal.with_new_columns(applied)
        return DataFrame(internal)
