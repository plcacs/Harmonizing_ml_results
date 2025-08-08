def get_dummies(data: Union[ArrayLike, Series, DataFrame],
                prefix: Optional[Union[str, List[str], Dict[str, str]]] = None,
                prefix_sep: str = '_',
                dummy_na: bool = False,
                columns: Optional[Iterable] = None,
                sparse: bool = False,
                drop_first: bool = False,
                dtype: Optional[Dtype] = None) -> DataFrame:
    ...

def _get_dummies_1d(data: Union[ArrayLike, Series],
                    prefix: Optional[Union[str, List[str], Dict[str, str]]] = None,
                    prefix_sep: str = '_',
                    dummy_na: bool = False,
                    sparse: bool = False,
                    drop_first: bool = False,
                    dtype: Optional[Dtype] = None) -> DataFrame:
    ...

def from_dummies(data: DataFrame,
                 sep: Optional[str] = None,
                 default_category: Optional[Union[Hashable, Dict[str, Hashable]]] = None) -> DataFrame:
    ...
