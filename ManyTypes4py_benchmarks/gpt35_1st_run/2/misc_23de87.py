def scatter_matrix(frame: DataFrame, alpha: float = 0.5, figsize: tuple = None, ax: Axes = None, grid: bool = False, diagonal: str = 'hist', marker: str = '.', density_kwds: dict = None, hist_kwds: dict = None, range_padding: float = 0.05, **kwds) -> Axes:

def radviz(frame: DataFrame, class_column: str, ax: Axes = None, color: str = None, colormap: str = None, **kwds) -> Axes:

def andrews_curves(frame: DataFrame, class_column: str, ax: Axes = None, samples: int = 200, color: str = None, colormap: str = None, **kwds) -> Axes:

def bootstrap_plot(series: Series, fig: Figure = None, size: int = 50, samples: int = 500, **kwds) -> Figure:

def parallel_coordinates(frame: DataFrame, class_column: str, cols: list = None, ax: Axes = None, color: str = None, use_columns: bool = False, xticks: list = None, colormap: str = None, axvlines: bool = True, axvlines_kwds: dict = None, sort_labels: bool = False, **kwds) -> Axes:

def lag_plot(series: Series, lag: int = 1, ax: Axes = None, **kwds) -> Axes:

def autocorrelation_plot(series: Series, ax: Axes = None, **kwds) -> Axes:

def unpack_single_str_list(keys: Union[str, List[str]]) -> str:
