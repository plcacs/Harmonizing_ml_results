def make_doc(name: str, ndim: int) -> str:
    """
    Generate the docstring for a Series/DataFrame reduction.
    """
    if ndim == 1:
        name1: str = 'scalar'
        name2: str = 'Series'
        axis_descr: str = '{index (0)}'
    else:
        name1 = 'Series'
        name2 = 'DataFrame'
        axis_descr = '{index (0), columns (1)}'
    if name == 'any':
        base_doc: str = _bool_doc
        desc: str = _any_desc
        see_also: str = _any_see_also
        examples: str = _any_examples
        kwargs: dict = {'empty_value': 'False'}
    elif name == 'all':
        base_doc = _bool_doc
        desc = _all_desc
        see_also = _all_see_also
        examples = _all_examples
        kwargs = {'empty_value': 'True'}
    elif name == 'min':
        base_doc = _num_doc
        desc = (
            "Return the minimum of the values over the requested axis.\n\n"
            "If you want the *index* of the minimum, use ``idxmin``. This is the equivalent "
            "of the ``numpy.ndarray`` method ``argmin``."
        )
        see_also = _stat_func_see_also
        examples = _min_examples
        kwargs = {'min_count': ''}
    elif name == 'max':
        base_doc = _num_doc
        desc = (
            "Return the maximum of the values over the requested axis.\n\n"
            "If you want the *index* of the maximum, use ``idxmax``. This is the equivalent "
            "of the ``numpy.ndarray`` method ``argmax``."
        )
        see_also = _stat_func_see_also
        examples = _max_examples
        kwargs = {'min_count': ''}
    elif name == 'sum':
        base_doc = _sum_prod_doc
        desc = (
            "Return the sum of the values over the requested axis.\n\n"
            "This is equivalent to the method ``numpy.sum``."
        )
        see_also = _stat_func_see_also
        examples = _sum_examples
        kwargs = {'min_count': _min_count_stub}
    elif name == 'prod':
        base_doc = _sum_prod_doc
        desc = "Return the product of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = _prod_examples
        kwargs = {'min_count': _min_count_stub}
    elif name == 'median':
        base_doc = _num_doc
        desc = "Return the median of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = (
            "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.median()\n            2.0\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, "
            "index=['tiger', 'zebra'])\n            >>> df\n            a   1\n            b   2\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.median(axis=1)\n            tiger   1.5\n            zebra   2.5\n            dtype: float64\n\n            In this case, "
            "`numeric_only` should be set to `True`\n            to avoid getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},\n            ...                   index=['tiger', 'zebra'])\n            >>> df.median(numeric_only=True)\n            a   1.5\n            dtype: float64"
        )
        kwargs = {'min_count': ''}
    elif name == 'mean':
        base_doc = _num_doc
        desc = "Return the mean of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = (
            "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.mean()\n            2.0\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, "
            "index=['tiger', 'zebra'])\n            >>> df.mean()\n            a   1.5\n            b   2.5\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.mean(axis=1)\n            tiger   1.5\n            zebra   2.5\n            dtype: float64\n\n            In this case, "
            "`numeric_only` should be set to `True` to avoid\n            getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},\n            ...                   index=['tiger', 'zebra'])\n            >>> df.mean(numeric_only=True)\n            a   1.5\n            dtype: float64"
        )
        kwargs = {'min_count': ''}
    elif name == 'var':
        base_doc = _num_ddof_doc
        desc = (
            "Return unbiased variance over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument."
        )
        examples = _var_examples
        see_also = ''
        kwargs = {'notes': ''}
    elif name == 'std':
        base_doc = _num_ddof_doc
        desc = (
            "Return sample standard deviation over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument."
        )
        examples = _std_examples
        see_also = _std_see_also.format(name2=name2)
        kwargs = {'notes': '', 'return_desc': _std_return_desc}
    elif name == 'sem':
        base_doc = _num_ddof_doc
        desc = (
            "Return unbiased standard error of the mean over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument"
        )
        examples = (
            "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.sem().round(6)\n            0.57735\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, "
            "index=['tiger', 'zebra'])\n            >>> df.sem()\n            a   0.5\n            b   0.5\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.sem(axis=1)\n            tiger   0.5\n            zebra   0.5\n            dtype: float64\n\n            In this case, "
            "`numeric_only` should be set to `True`\n            to avoid getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},\n            ...                   index=['tiger', 'zebra'])\n            >>> df.sem(numeric_only=True)\n            a   0.5\n            dtype: float64"
        )
        see_also = _sem_see_also.format(name2=name2)
        kwargs = {'notes': '', 'return_desc': _sem_return_desc}
    elif name == 'skew':
        base_doc = _num_doc
        desc = "Return unbiased skew over requested axis.\n\nNormalized by N-1."
        see_also = _skew_see_also
        examples = (
            "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.skew()\n            0.0\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 3, 5]},\n            ...                   index=['tiger', 'zebra', 'cow'])\n            >>> df.skew()\n            a   0.0\n            b   0.0\n            c   0.0\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.skew(axis=1)\n            tiger   1.732051\n            zebra  -1.732051\n            cow     0.000000\n            dtype: float64\n\n            In this case, `numeric_only` should be set to `True` to avoid\n            getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['T', 'Z', 'X']},\n            ...                   index=['tiger', 'zebra', 'cow'])\n            >>> df.skew(numeric_only=True)\n            a   0.0\n            dtype: float64"
        )
        kwargs = {'min_count': ''}
    elif name == 'kurt':
        base_doc = _num_doc
        desc = (
            "Return unbiased kurtosis over requested axis.\n\nKurtosis obtained using Fisher's definition of\nkurtosis (kurtosis of normal == 0.0). Normalized by N-1."
        )
        see_also = ''
        examples = (
            "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 2, 3], index=['cat', 'dog', 'dog', 'mouse'])\n            >>> s.kurt()\n            1.5\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},\n            ...                   index=['cat', 'dog', 'dog', 'mouse'])\n            >>> df.kurt()\n            a   1.5\n            b   4.0\n            dtype: float64\n\n            With axis=None\n\n            >>> df.kurt(axis=None).round(6)\n            -0.988693\n\n            Using axis=1\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},\n            ...                   index=['cat', 'dog'])\n            >>> df.kurt(axis=1)\n            cat   -6.0\n            dog   -6.0\n            dtype: float64"
        )
        kwargs = {'min_count': ''}
    elif name == 'cumsum':
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = 'sum'
        see_also = ''
        examples = _cumsum_examples
        kwargs = {'accum_func_name': 'sum'}
    elif name == 'cumprod':
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = 'product'
        see_also = ''
        examples = _cumprod_examples
        kwargs = {'accum_func_name': 'prod'}
    elif name == 'cummin':
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = 'minimum'
        see_also = ''
        examples = _cummin_examples
        kwargs = {'accum_func_name': 'min'}
    elif name == 'cummax':
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = 'maximum'
        see_also = ''
        examples = _cummax_examples
        kwargs = {'accum_func_name': 'max'}
    else:
        raise NotImplementedError
    docstr: str = base_doc.format(
        desc=desc,
        name=name,
        name1=name1,
        name2=name2,
        axis_descr=axis_descr,
        see_also=see_also,
        examples=examples,
        **kwargs
    )
    return docstr