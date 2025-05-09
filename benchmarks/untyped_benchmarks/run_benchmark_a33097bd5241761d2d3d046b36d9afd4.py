
'\nBenchmark comprehensions.\n\nAuthor: Carl Meyer\n'
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional
import pyperf

class WidgetKind(Enum):
    BIG = 1
    SMALL = 2

@dataclass
class Widget():
    widget_id = None
    creator_id = None
    derived_widget_ids = None
    kind = None
    has_knob = None
    has_spinner = None

class WidgetTray():

    def __init__(self, owner_id, widgets):
        self.owner_id = owner_id
        self.sorted_widgets = []
        self._add_widgets(widgets)

    def _any_knobby(self, widgets):
        return any((w.has_knob for w in widgets if w))

    def _is_big_spinny(self, widget):
        return ((widget.kind == WidgetKind.BIG) and widget.has_spinner)

    def _add_widgets(self, widgets):
        widgets = [w for w in widgets if (not self._is_big_spinny(w))]
        id_to_widget = {w.widget_id: w for w in widgets}
        id_to_derived = {w.widget_id: [id_to_widget.get(dwid) for dwid in w.derived_widget_ids] for w in widgets}
        sortable_widgets = [((w.creator_id == self.owner_id), self._any_knobby(id_to_derived[w.widget_id]), len(id_to_derived[w.widget_id]), w.widget_id) for w in widgets]
        sortable_widgets.sort()
        self.sorted_widgets = [id_to_widget[sw[(- 1)]] for sw in sortable_widgets]

def make_some_widgets():
    widget_id = 0
    widgets = []
    for creator_id in range(3):
        for kind in WidgetKind:
            for has_knob in [True, False]:
                for has_spinner in [True, False]:
                    derived = [w.widget_id for w in widgets[::(creator_id + 1)]]
                    widgets.append(Widget(widget_id, creator_id, derived, kind, has_knob, has_spinner))
                    widget_id += 1
    assert (len(widgets) == 24)
    return widgets

def bench_comprehensions(loops):
    range_it = range(loops)
    widgets = make_some_widgets()
    t0 = pyperf.perf_counter()
    for _ in range_it:
        tray = WidgetTray(1, widgets)
        assert (len(tray.sorted_widgets) == 18)
    return (pyperf.perf_counter() - t0)
if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark comprehensions'
    runner.bench_time_func('comprehensions', bench_comprehensions)
