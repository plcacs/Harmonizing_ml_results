import warnings
import numpy as np
from pandas import NA, Categorical, DataFrame, Index, Series
from pandas.arrays import StringArray

class Dtypes:
    params: list[str] = ['str', 'string[python]', 'string[pyarrow]']
    param_names: list[str] = ['dtype']

    def setup(self, dtype: str) -> None:
        try:
            self.s: Series = Series(Index([f'i-{i}' for i in range(10000)], dtype=object)._values, dtype=dtype)
        except ImportError as err:
            raise NotImplementedError from err

class Construction:
    params: tuple[list[str], list[str]] = (['series', 'frame', 'categorical_series'], ['str', 'string[python]', 'string[pyarrow]'])
    param_names: list[str] = ['pd_type', 'dtype']
    pd_mapping: dict[str, type] = {'series': Series, 'frame': DataFrame, 'categorical_series': Series}
    dtype_mapping: dict[str, type] = {'str': str, 'string[python]': object, 'string[pyarrow]': object}

    def setup(self, pd_type: str, dtype: str) -> None:
        series_arr: np.ndarray = np.array([str(i) * 10 for i in range(100000)], dtype=self.dtype_mapping[dtype])
        if pd_type == 'series':
            self.arr: np.ndarray = series_arr
        elif pd_type == 'frame':
            self.arr: np.ndarray = series_arr.reshape((50000, 2)).copy()
        elif pd_type == 'categorical_series':
            self.arr: Categorical = Categorical(series_arr)

    def time_construction(self, pd_type: str, dtype: str) -> None:
        self.pd_mapping[pd_type](self.arr, dtype=dtype)

    def peakmem_construction(self, pd_type: str, dtype: str) -> None:
        self.pd_mapping[pd_type](self.arr, dtype=dtype)

class Methods(Dtypes):

    def time_center(self, dtype: str) -> None:
        self.s.str.center(100)

    def time_count(self, dtype: str) -> None:
        self.s.str.count('A')

    # Add other methods with type annotations

class Repeat:
    params: list[str] = ['int', 'array']
    param_names: list[str] = ['repeats']

    def setup(self, repeats: str) -> None:
        N: int = 10 ** 5
        self.s: Series = Series(Index([f'i-{i}' for i in range(N)], dtype=object))
        repeat: dict[str, any] = {'int': 1, 'array': np.random.randint(1, 3, N)}
        self.values: any = repeat[repeats]

    def time_repeat(self, repeats: str) -> None:
        self.s.str.repeat(self.values)

# Add other classes with type annotations
