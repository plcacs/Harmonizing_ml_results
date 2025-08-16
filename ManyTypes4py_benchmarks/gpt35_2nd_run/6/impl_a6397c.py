def dtype_for_elements_strategy(s: st.SearchStrategy) -> st.SearchStrategy:
def infer_dtype_if_necessary(dtype: Optional[Any], values: Any, elements: st.SearchStrategy, draw: Any) -> Any:
def elements_and_dtype(elements: Optional[st.SearchStrategy], dtype: Optional[Any], source: Optional[str] = None) -> Tuple[st.SearchStrategy, Any]:
class ValueIndexStrategy(st.SearchStrategy):
    def __init__(self, elements: st.SearchStrategy, dtype: Any, min_size: int, max_size: int, unique: bool, name: str):
    def do_draw(self, data: Any) -> Any:
@cacheable
@defines_strategy()
def range_indexes(min_size: int = 0, max_size: Optional[int] = None, name: Optional[st.SearchStrategy] = st.none()) -> st.SearchStrategy:
@cacheable
@defines_strategy()
def indexes(*, elements: Optional[st.SearchStrategy] = None, dtype: Optional[Any] = None, min_size: int = 0, max_size: Optional[int] = None, unique: bool = True, name: Optional[st.SearchStrategy] = st.none()) -> st.SearchStrategy:
@defines_strategy()
def series(*, elements: Optional[st.SearchStrategy] = None, dtype: Optional[Any] = None, index: Optional[st.SearchStrategy] = None, fill: Optional[Any] = None, unique: bool = False, name: Optional[st.SearchStrategy] = st.none()) -> st.SearchStrategy:
@attr.s(slots=True)
class column(Generic[Ex]):
    def __init__(self, name: Optional[Any] = None, elements: Optional[st.SearchStrategy] = None, dtype: Optional[Any] = None, fill: Optional[Any] = None, unique: bool = False):
def columns(names_or_number, *, dtype: Optional[Any] = None, elements: Optional[st.SearchStrategy] = None, fill: Optional[Any] = None, unique: bool = False) -> List[column]:
@defines_strategy()
def data_frames(columns: Optional[Union[List[column], int]] = None, *, rows: Optional[st.SearchStrategy] = None, index: Optional[st.SearchStrategy] = None) -> st.SearchStrategy:
