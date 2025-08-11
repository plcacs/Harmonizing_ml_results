import warnings
import numpy as np
from pandas import NA, Categorical, DataFrame, Index, Series
from pandas.arrays import StringArray

class Dtypes:
    params = ['str', 'string[python]', 'string[pyarrow]']
    param_names = ['dtype']

    def setup(self, dtype) -> None:
        try:
            self.s = Series(Index([f'i-{i}' for i in range(10000)], dtype=object)._values, dtype=dtype)
        except ImportError as err:
            raise NotImplementedError from err

class Construction:
    params = (['series', 'frame', 'categorical_series'], ['str', 'string[python]', 'string[pyarrow]'])
    param_names = ['pd_type', 'dtype']
    pd_mapping = {'series': Series, 'frame': DataFrame, 'categorical_series': Series}
    dtype_mapping = {'str': 'str', 'string[python]': object, 'string[pyarrow]': object}

    def setup(self, pd_type, dtype) -> None:
        series_arr = np.array([str(i) * 10 for i in range(100000)], dtype=self.dtype_mapping[dtype])
        if pd_type == 'series':
            self.arr = series_arr
        elif pd_type == 'frame':
            self.arr = series_arr.reshape((50000, 2)).copy()
        elif pd_type == 'categorical_series':
            self.arr = Categorical(series_arr)

    def time_construction(self, pd_type: Union[bool, numpy.ndarray], dtype: Union[bool, numpy.ndarray]) -> None:
        self.pd_mapping[pd_type](self.arr, dtype=dtype)

    def peakmem_construction(self, pd_type: Union[bool, str, numpy.ndarray], dtype: Union[bool, str, numpy.ndarray]) -> None:
        self.pd_mapping[pd_type](self.arr, dtype=dtype)

class Methods(Dtypes):

    def time_center(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.center(100)

    def time_count(self, dtype: Union[bool, typing.Type, str, None]) -> None:
        self.s.str.count('A')

    def time_endswith(self, dtype: Union[bool, typing.Type, str, None]) -> None:
        self.s.str.endswith('A')

    def time_extract(self, dtype: Union[bool, typing.Type, str, None]) -> None:
        with warnings.catch_warnings(record=True):
            self.s.str.extract('(\\w*)A(\\w*)')

    def time_findall(self, dtype: Union[typing.Type, bool, str, None]) -> None:
        self.s.str.findall('[A-Z]+')

    def time_find(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.find('[A-Z]+')

    def time_rfind(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.rfind('[A-Z]+')

    def time_fullmatch(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.fullmatch('A')

    def time_get(self, dtype: Union[bool, typing.Type]) -> None:
        self.s.str.get(0)

    def time_len(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.len()

    def time_join(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.join(' ')

    def time_match(self, dtype: Union[bool, typing.Type]) -> None:
        self.s.str.match('A')

    def time_normalize(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.normalize('NFC')

    def time_pad(self, dtype: Union[typing.Type, typing.Callable[None, bool], bool]) -> None:
        self.s.str.pad(100, side='both')

    def time_partition(self, dtype: bool) -> None:
        self.s.str.partition('A')

    def time_rpartition(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.rpartition('A')

    def time_replace(self, dtype: Union[typing.Type, bool]) -> None:
        self.s.str.replace('A', '\x01\x01')

    def time_translate(self, dtype: Union[bool, typing.Type, str, None]) -> None:
        self.s.str.translate({'A': '\x01\x01'})

    def time_slice(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.slice(5, 15, 2)

    def time_startswith(self, dtype: Union[bool, typing.Type, str, None]) -> None:
        self.s.str.startswith('A')

    def time_strip(self, dtype: Union[bool, typing.Type]) -> None:
        self.s.str.strip('A')

    def time_rstrip(self, dtype: Union[typing.Type, bool, str, None]) -> None:
        self.s.str.rstrip('A')

    def time_lstrip(self, dtype: Union[bool, typing.Type]) -> None:
        self.s.str.lstrip('A')

    def time_title(self, dtype: Union[bool, typing.Type]) -> None:
        self.s.str.title()

    def time_upper(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.upper()

    def time_lower(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.lower()

    def time_wrap(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.wrap(10)

    def time_zfill(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.zfill(10)

    def time_isalnum(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.isalnum()

    def time_isalpha(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.isalpha()

    def time_isdecimal(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.isdecimal()

    def time_isdigit(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.isdigit()

    def time_islower(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.islower()

    def time_isnumeric(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.isnumeric()

    def time_isspace(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.isspace()

    def time_istitle(self, dtype: Union[typing.Type, bool, typing.Callable[None, bool]]) -> None:
        self.s.str.istitle()

    def time_isupper(self, dtype: Union[bool, typing.Type, typing.Callable[None, bool]]) -> None:
        self.s.str.isupper()

class Repeat:
    params = ['int', 'array']
    param_names = ['repeats']

    def setup(self, repeats) -> None:
        N = 10 ** 5
        self.s = Series(Index([f'i-{i}' for i in range(N)], dtype=object))
        repeat = {'int': 1, 'array': np.random.randint(1, 3, N)}
        self.values = repeat[repeats]

    def time_repeat(self, repeats: Union[typing.Iterable[int], int, typing.Hashable]) -> None:
        self.s.str.repeat(self.values)

class Cat:
    params = ([0, 3], [None, ','], [None, '-'], [0.0, 0.001, 0.15])
    param_names = ['other_cols', 'sep', 'na_rep', 'na_frac']

    def setup(self, other_cols, sep, na_rep, na_frac) -> None:
        N = 10 ** 5
        mask_gen = lambda: np.random.choice([True, False], N, p=[1 - na_frac, na_frac])
        self.s = Series(Index([f'i-{i}' for i in range(N)], dtype=object)).where(mask_gen())
        if other_cols == 0:
            self.others = None
        else:
            self.others = DataFrame({i: Index([f'i-{i}' for i in range(N)], dtype=object).where(mask_gen()) for i in range(other_cols)})

    def time_cat(self, other_cols: Union[bool, str], sep: Union[bool, str], na_rep: Union[bool, str], na_frac: Union[bool, str]) -> None:
        self.s.str.cat(others=self.others, sep=sep, na_rep=na_rep)

class Contains(Dtypes):
    params = (Dtypes.params, [True, False])
    param_names = ['dtype', 'regex']

    def setup(self, dtype, regex) -> None:
        super().setup(dtype)

    def time_contains(self, dtype: Union[str, None, typing.Iterable[typing.Any]], regex: Union[str, None, typing.Iterable]) -> None:
        self.s.str.contains('A', regex=regex)

class Split(Dtypes):
    params = (Dtypes.params, [True, False])
    param_names = ['dtype', 'expand']

    def setup(self, dtype, expand) -> None:
        super().setup(dtype)
        self.s = self.s.str.join('--')

    def time_split(self, dtype: Union[bool, str], expand: Union[str, bool, list[str]]) -> None:
        self.s.str.split('--', expand=expand)

    def time_rsplit(self, dtype: bool, expand: Union[str, bool, list[str]]) -> None:
        self.s.str.rsplit('--', expand=expand)

class Extract(Dtypes):
    params = (Dtypes.params, [True, False])
    param_names = ['dtype', 'expand']

    def setup(self, dtype, expand) -> None:
        super().setup(dtype)

    def time_extract_single_group(self, dtype: bool, expand: Union[bool, str, typing.Callable[int,int, None]]) -> None:
        with warnings.catch_warnings(record=True):
            self.s.str.extract('(\\w*)A', expand=expand)

class Dummies(Dtypes):

    def setup(self, dtype) -> None:
        super().setup(dtype)
        N = len(self.s) // 5
        self.s = self.s[:N].str.join('|')

    def time_get_dummies(self, dtype: Union[numpy.ndarray, None, bool]) -> None:
        self.s.str.get_dummies('|')

class Encode:

    def setup(self) -> None:
        self.ser = Series(Index([f'i-{i}' for i in range(10000)], dtype=object))

    def time_encode_decode(self) -> None:
        self.ser.str.encode('utf-8').str.decode('utf-8')

class Slice:

    def setup(self) -> None:
        self.s = Series(['abcdefg', np.nan] * 500000)

    def time_vector_slice(self) -> None:
        self.s.str[:5]

class Iter(Dtypes):

    def time_iter(self, dtype: Union[bool, typing.Type]) -> None:
        for i in self.s:
            pass

class StringArrayConstruction:

    def setup(self) -> None:
        self.series_arr = np.array([str(i) * 10 for i in range(10 ** 5)], dtype=object)
        self.series_arr_nan = np.concatenate([self.series_arr, np.array([NA] * 1000)])

    def time_string_array_construction(self) -> None:
        StringArray(self.series_arr)

    def time_string_array_with_nan_construction(self) -> None:
        StringArray(self.series_arr_nan)

    def peakmem_stringarray_construction(self) -> None:
        StringArray(self.series_arr)