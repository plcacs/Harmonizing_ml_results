from __future__ import annotations
from typing import TYPE_CHECKING, Any, Final, cast

class HTMLFormatter:
    indent_delta: Final[int] = 2

    def __init__(self, formatter: Any, classes: Any = None, border: Any = None, table_id: Any = None, render_links: bool = False) -> None:
        self.fmt = formatter
        self.classes = classes
        self.frame = self.fmt.frame
        self.columns = self.fmt.tr_frame.columns
        self.elements: list[str] = []
        self.bold_rows = self.fmt.bold_rows
        self.escape = self.fmt.escape
        self.show_dimensions = self.fmt.show_dimensions
        self.border: int | None
        self.table_id = table_id
        self.render_links = render_links
        self.col_space: dict[Any, Any] = {}

    def to_string(self) -> str:
        lines = self.render()
        if any((isinstance(x, str) for x in lines)):
            lines = [str(x) for x in lines]
        return '\n'.join(lines)

    def render(self) -> list[str]:
        self._write_table()
        if self.should_show_dimensions:
            by = chr(215)
            self.write(f'<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>')
        return self.elements

    @property
    def should_show_dimensions(self) -> bool:
        return self.fmt.should_show_dimensions

    @property
    def show_row_idx_names(self) -> Any:
        return self.fmt.show_row_idx_names

    @property
    def show_col_idx_names(self) -> Any:
        return self.fmt.show_col_idx_names

    @property
    def row_levels(self) -> int:
        return 0

    def _get_columns_formatted_values(self) -> Any:
        return self.columns

    @property
    def is_truncated(self) -> bool:
        return self.fmt.is_truncated

    @property
    def ncols(self) -> int:
        return len(self.fmt.tr_frame.columns)

    def write(self, s: Any, indent: int = 0) -> None:
        rs = pprint_thing(s)
        self.elements.append(' ' * indent + rs)

    def write_th(self, s: Any, header: bool = False, indent: int = 0, tags: Any = None) -> None:
        ...

    def write_td(self, s: Any, indent: int = 0, tags: Any = None) -> None:
        ...

    def _write_cell(self, s: Any, kind: str = 'td', indent: int = 0, tags: Any = None) -> None:
        ...

    def write_tr(self, line: Any, indent: int = 0, indent_delta: int = 0, header: bool = False, align: Any = None, tags: Any = None, nindex_levels: int = 0) -> None:
        ...

    def _write_table(self, indent: int = 0) -> None:
        ...

    def _write_col_header(self, indent: int) -> None:
        ...

    def _write_row_header(self, indent: int) -> None:
        ...

    def _write_header(self, indent: int) -> None:
        ...

    def _get_formatted_values(self) -> Any:
        ...

    def _write_body(self, indent: int) -> None:
        ...

    def _write_regular_rows(self, fmt_values: Any, indent: int) -> None:
        ...

    def _write_hierarchical_rows(self, fmt_values: Any, indent: int) -> None:
        ...

class NotebookFormatter(HTMLFormatter):
    def _get_formatted_values(self) -> Any:
        ...

    def _get_columns_formatted_values(self) -> Any:
        ...

    def write_style(self) -> None:
        ...

    def render(self) -> list[str]:
        ...
