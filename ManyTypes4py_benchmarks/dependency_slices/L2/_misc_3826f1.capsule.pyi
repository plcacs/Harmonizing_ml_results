from typing import Any

# === Third-party dependency: matplotlib.axes ===
# Used symbols: Axes

# === Third-party dependency: matplotlib.figure ===
class Figure(FigureBase): ...

# === Third-party dependency: matplotlib.table ===
class Table(Artist): ...

# === Third-party dependency: numpy ===
# Used symbols: ndarray

# === Internal dependency: pandas.plotting._core ===
def _get_plot_backend(backend: str | None = ...) -> Any: ...