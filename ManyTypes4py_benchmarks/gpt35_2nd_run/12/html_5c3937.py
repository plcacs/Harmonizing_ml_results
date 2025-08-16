from __future__ import annotations
from typing import TYPE_CHECKING, Any, Final, cast

class HTMLFormatter:
    indent_delta: Final[int] = 2

    def __init__(self, formatter, classes=None, border=None, table_id=None, render_links=False) -> None:
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
        self.col_space: dict[str, str] = {}

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
    def show_row_idx_names(self) -> bool:
        return self.fmt.show_row_idx_names

    @property
    def show_col_idx_names(self) -> bool:
        return self.fmt.show_col_idx_names

    @property
    def row_levels(self) -> int:
        return self.fmt.index

    def _get_columns_formatted_values(self) -> Any:
        return self.columns

    @property
    def is_truncated(self) -> bool:
        return self.fmt.is_truncated

    @property
    def ncols(self) -> int:
        return len(self.fmt.tr_frame.columns)

    def write(self, s, indent=0) -> None:
        rs = pprint_thing(s)
        self.elements.append(' ' * indent + rs)

    def write_th(self, s, header=False, indent=0, tags=None) -> None:
        ...

    def write_td(self, s, indent=0, tags=None) -> None:
        ...

    def _write_cell(self, s, kind='td', indent=0, tags=None) -> None:
        ...

    def write_tr(self, line, indent=0, indent_delta=0, header=False, align=None, tags=None, nindex_levels=0) -> None:
        ...

    def _write_table(self, indent=0) -> None:
        ...

    def _write_col_header(self, indent) -> None:
        ...

    def _write_row_header(self, indent) -> None:
        ...

    def _write_header(self, indent) -> None:
        ...

    def _get_formatted_values(self) -> dict[int, Any]:
        ...

    def _write_body(self, indent) -> None:
        ...

    def _write_regular_rows(self, fmt_values, indent) -> None:
        ...

    def _write_hierarchical_rows(self, fmt_values, indent) -> None:
        ...

class NotebookFormatter(HTMLFormatter):
    def _get_formatted_values(self) -> dict[int, Any]:
        ...

    def _get_columns_formatted_values(self) -> Any:
        ...

    def write_style(self) -> None:
        ...

    def render(self) -> list[str]:
        ...
