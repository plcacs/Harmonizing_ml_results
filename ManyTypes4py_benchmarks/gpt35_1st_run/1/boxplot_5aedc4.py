def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs: dict) -> None:
    ...

class BoxPlot(LinePlot):
    _valid_return_types: tuple[Literal[None, 'axes', 'dict', 'both']] = (None, 'axes', 'dict', 'both')

    class BP(NamedTuple):
        ax: Axes
        lines: Line2D

    def __init__(self, data: Collection, return_type: Literal[None, 'axes', 'dict', 'both'] = 'axes', **kwargs: dict) -> None:
        ...

    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, column_num: int = None, return_type: Literal[None, 'axes', 'dict', 'both'] = 'axes', **kwds: dict) -> tuple[Axes, Line2D]:
        ...

    def _validate_color_args(self, color: dict | None, colormap: MatplotlibColor | None) -> dict | None:
        ...

    def maybe_color_bp(self, bp: dict) -> None:
        ...

    def _make_plot(self, fig: Figure) -> None:
        ...

    def _make_legend(self) -> None:
        ...

    def _post_plot_logic(self, ax: Axes, data: pd.DataFrame) -> None:
        ...

    @property
    def orientation(self) -> Literal['vertical', 'horizontal']:
        ...

    @property
    def result(self) -> pd.Series | object:
        ...

def maybe_color_bp(bp: dict, color_tup: tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor, MatplotlibColor], **kwds: dict) -> None:
    ...

def _grouped_plot_by_column(plotf: callable, data: pd.DataFrame, columns: list[str] | None = None, by: str | list[str] | None = None, numeric_only: bool = True, grid: bool = False, figsize: tuple[int, int] | None = None, ax: Axes | None = None, layout: tuple[int, int] | None = None, return_type: Literal[None, 'axes'] | None = None, **kwargs: dict) -> pd.Series | tuple[Axes]:
    ...

def boxplot(data: pd.DataFrame | ABCSeries, column: str | list[str] | None = None, by: str | list[str] | None = None, ax: Axes | None = None, fontsize: int | None = None, rot: int = 0, grid: bool = True, figsize: tuple[int, int] | None = None, layout: tuple[int, int] | None = None, return_type: Literal['axes', 'dict', 'both'] | None = None, **kwds: dict) -> pd.Series | Axes | dict:
    ...

def boxplot_frame(self, column: str | list[str] | None = None, by: str | list[str] | None = None, ax: Axes | None = None, fontsize: int | None = None, rot: int = 0, grid: bool = True, figsize: tuple[int, int] | None = None, layout: tuple[int, int] | None = None, return_type: Literal['axes', 'dict', 'both'] | None = None, **kwds: dict) -> Axes | dict:
    ...

def boxplot_frame_groupby(grouped: pd.DataFrame, subplots: bool = True, column: str | list[str] | None = None, fontsize: int | None = None, rot: int = 0, grid: bool = True, ax: Axes | None = None, figsize: tuple[int, int] | None = None, layout: tuple[int, int] | None = None, sharex: bool = False, sharey: bool = True, **kwds: dict) -> pd.Series | Axes:
    ...
