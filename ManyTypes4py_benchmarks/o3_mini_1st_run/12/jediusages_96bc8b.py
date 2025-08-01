#!/usr/bin/env python3
"""Common jedi class to work with Jedi functions
"""
import sublime
from functools import partial
from ._typing import Union, Dict
from typing import Any, Optional, List, Tuple

class JediUsages(object):
    """Work with Jedi definitions
    """

    def __init__(self, text: Any) -> None:
        self.text: Any = text
        self.options: Any = None  # type: Union[str, List[Tuple[Any, ...]]]
        self.point: Any = None

    def process(self, usages: bool = False, data: Optional[Dict[str, Any]] = None) -> None:
        """Process the definitions
        """
        # Assuming data is provided and is a dict
        assert data is not None, "data parameter is required"
        view = self.text.view
        if not data['success']:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        definitions: Any = data['goto'] if not usages else data['usages']
        if len(definitions) == 0:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        if definitions is not None and len(definitions) == 1 and (not usages):
            self._jump(*definitions[0])
        else:
            self._show_options(definitions, usages)

    def _jump(self, filename: Union[int, str], lineno: Optional[int] = None, columno: Optional[int] = None, transient: bool = False) -> None:
        """Jump to a window
        """
        if isinstance(filename, int):
            if filename == -1:
                view = self.text.view
                point = self.point
                sublime.active_window().focus_view(view)
                view.show(point)
                if view.sel()[0] != point:
                    view.sel().clear()
                    view.sel().add(point)
                return
        opts: Any = self.options[filename]
        if len(self.options[filename]) == 4:
            opts = opts[1:]
        filename, lineno, columno = opts  # type: ignore
        flags: int = sublime.ENCODED_POSITION
        if transient:
            flags |= sublime.TRANSIENT
        sublime.active_window().open_file('{}:{}:{}'.format(filename, lineno or 0, columno or 0), flags)
        self._toggle_indicator(lineno, columno)

    def _show_options(self, defs: Union[str, List[Tuple[Any, ...]]], usages: bool) -> None:
        """Show a dropdown quickpanel with options to jump
        """
        view = self.text.view
        if usages or (not usages and not isinstance(defs, str)):
            if len(defs) == 4:
                options: List[List[Any]] = [[o[0], o[1], 'line: {} column: {}'.format(o[2], o[3])] for o in defs]  # type: ignore
            else:
                options = [[o[0], 'line: {} column: {}'.format(o[1], o[2])] for o in defs]  # type: ignore
        elif len(defs):
            options = defs[0]  # type: ignore
        else:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        self.options = defs
        self.point = self.text.view.sel()[0]
        self.text.view.window().show_quick_panel(options, self._jump, on_highlight=partial(self._jump, transient=True))

    def _toggle_indicator(self, lineno: int = 0, columno: int = 0) -> None:
        """Toggle mark indicator for focus the cursor
        """
        pt: int = self.text.view.text_point(lineno - 1, columno)
        region_name: str = 'anaconda.indicator.{}.{}'.format(self.text.view.id(), lineno)
        for i in range(3):
            delta: int = 300 * i * 2
            sublime.set_timeout(lambda: self.text.view.add_regions(
                region_name,
                [sublime.Region(pt, pt)],
                'comment',
                'bookmark',
                sublime.DRAW_EMPTY_AS_OVERWRITE
            ), delta)
            sublime.set_timeout(lambda: self.text.view.erase_regions(region_name), delta + 300)
