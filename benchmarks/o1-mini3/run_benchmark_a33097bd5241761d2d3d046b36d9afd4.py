from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Dict, Tuple
import pyperf

class WidgetKind(Enum):
    BIG = 1
    SMALL = 2

@dataclass
class Widget:
    widget_id: int
    creator_id: int
    derived_widget_ids: List[int]
    kind: WidgetKind
    has_knob: bool
    has_spinner: bool

class WidgetTray:

    owner_id: int
    sorted_widgets: List[Widget]

    def __init__(self, owner_id: int, widgets: List[Widget]) -> None:
        self.owner_id = owner_id
        self.sorted_widgets = []
        self._add_widgets(widgets)

    def _any_knobby(self, widgets: List[Optional[Widget]]) -> bool:
        return any(w.has_knob for w in widgets if w)

    def _is_big_spinny(self, widget: Widget) -> bool:
        return (widget.kind == WidgetKind.BIG) and widget.has_spinner

    def _add_widgets(self, widgets: List[Widget]) -> None:
        widgets = [w for w in widgets if not self._is_big_spinny(w)]
        id_to_widget: Dict[int, Widget] = {w.widget_id: w for w in widgets}
        id_to_derived: Dict[int, List[Optional[Widget]]] = {
            w.widget_id: [id_to_widget.get(dwid) for dwid in w.derived_widget_ids]
            for w in widgets
        }
        sortable_widgets: List[Tuple[bool, bool, int, int]] = [
            (
                (w.creator_id == self.owner_id),
                self._any_knobby(id_to_derived[w.widget_id]),
                len(id_to_derived[w.widget_id]),
                w.widget_id
            )
            for w in widgets
        ]
        sortable_widgets.sort()
        self.sorted_widgets = [id_to_widget[sw[3]] for sw in sortable_widgets]

def make_some_widgets() -> List[Widget]:
    widget_id: int = 0
    widgets: List[Widget] = []
    for creator_id in range(3):
        for kind in WidgetKind:
            for has_knob in [True, False]:
                for has_spinner in [True, False]:
                    derived: List[int] = [w.widget_id for w in widgets[::(creator_id + 1)]]
                    widgets.append(Widget(widget_id, creator_id, derived, kind, has_knob, has_spinner))
                    widget_id += 1
    assert len(widgets) == 24
    return widgets

def bench_comprehensions(loops: int) -> float:
    range_it = range(loops)
    widgets = make_some_widgets()
    t0 = pyperf.perf_counter()
    for _ in range_it:
        tray = WidgetTray(1, widgets)
        assert len(tray.sorted_widgets) == 18
    return pyperf.perf_counter() - t0

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark comprehensions'
    runner.bench_time_func('comprehensions', bench_comprehensions)
