"""Common jedi class to work with Jedi functions
"""
import sublime
from functools import partial
from ._typing import Union, Dict
from typing import Any, Optional, Sequence, Tuple, Union as TypingUnion

Location3 = Tuple[str, int, int]
Location4 = Tuple[str, str, int, int]
DefsType = Sequence[TypingUnion[Location3, Location4]]


class JediUsages(object):
    """Work with Jedi definitions
    """

    def __init__(self, text: Any) -> None:
        self.text: Any = text
        self.options: Optional[TypingUnion[DefsType, str]] = None
        self.point: Optional[sublime.Region] = None

    def process(self, usages: bool = False, data: Optional[Dict[str, Any]] = None) -> None:
        """Process the definitions
        """
        view = self.text.view
        if not data or not data['success']:
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

    def _jump(
        self,
        filename: TypingUnion[int, str],
        lineno: Optional[int] = None,
        columno: Optional[int] = None,
        transient: bool = False
    ) -> None:
        """Jump to a window
        """
        if type(filename) is int:
            if filename == -1:
                view = self.text.view
                point = self.point
                sublime.active_window().focus_view(view)
                view.show(point)
                if view.sel()[0] != point:
                    view.sel().clear()
                    view.sel().add(point)
                return
        opts = self.options[filename]  # type: ignore[index]
        if len(self.options[filename]) == 4:  # type: ignore[index]
            opts = opts[1:]
        filename, lineno, columno = opts  # type: ignore[misc]
        flags = sublime.ENCODED_POSITION
        if transient:
            flags |= sublime.TRANSIENT
        sublime.active_window().open_file('{}:{}:{}'.format(filename, lineno or 0, columno or 0), flags)
        self._toggle_indicator(lineno, columno)

    def _show_options(self, defs: TypingUnion[DefsType, str], usages: bool) -> None:
        """Show a dropdown quickpanel with options to jump
        """
        view = self.text.view
        if usages or (not usages and type(defs) is not str):
            if len(defs) == 4:  # type: ignore[arg-type]
                options = [[o[0], o[1], 'line: {} column: {}'.format(o[2], o[3])] for o in defs]  # type: ignore[index]
            else:
                options = [[o[0], 'line: {} column: {}'.format(o[1], o[2])] for o in defs]  # type: ignore[index]
        elif len(defs):  # type: ignore[arg-type]
            options = defs[0]  # type: ignore[index]
        else:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        self.options = defs
        self.point = self.text.view.sel()[0]
        self.text.view.window().show_quick_panel(options, self._jump, on_highlight=partial(self._jump, transient=True))

    def _toggle_indicator(self, lineno: int = 0, columno: int = 0) -> None:
        """Toggle mark indicator for focus the cursor
        """
        pt = self.text.view.text_point(lineno - 1, columno)
        region_name = 'anaconda.indicator.{}.{}'.format(self.text.view.id(), lineno)
        for i in range(3):
            delta = 300 * i * 2
            sublime.set_timeout(lambda: self.text.view.add_regions(region_name, [sublime.Region(pt, pt)], 'comment', 'bookmark', sublime.DRAW_EMPTY_AS_OVERWRITE), delta)
            sublime.set_timeout(lambda: self.text.view.erase_regions(region_name), delta + 300)