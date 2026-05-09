from typing import Union, List, Dict, Tuple, Any

class KoalasPlot:
    ...

def plot_koalas(data: Union[pd.Series, pd.DataFrame], kind: str, **kwargs) -> Any:
    ...

def plot_frame(data: pd.DataFrame, x: str, y: str, kind: str, **kwargs) -> Any:
    ...

_plot_klass: Dict[str, type] = {}
_plot_klass = {getattr(klass, '_kind'): klass for klass in _klasses}

def _plot(data: Union[pd.Series, pd.DataFrame], x: str, y: str, subplots: bool, ax: Any, kind: str, **kwargs) -> Any:
    ...
