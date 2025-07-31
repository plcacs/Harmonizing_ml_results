#!/usr/bin/env python3
"""
Common jedi class to work with Jedi functions
"""
import sublime
from functools import partial
from typing import Any, Dict, List, Tuple, Union, Optional
from ._typing import Union as _Union, Dict as _Dict  # Keeping original if needed

# Type aliases for definitions
Definition3 = Tuple[str, int, int]  # (display, lineno, columno)
Definition4 = Tuple[str, Any, int, int]  # (ignored?, filename, lineno, columno)
Definition = Union[Definition3, Definition4]
Definitions = List[Definition]

class JediUsages(object):
    """
    Work with Jedi definitions.
    """

    def __init__(self, text: Any) -> None:
        self.text: Any = text
        self.options: Optional[Union[Definitions, str]] = None
        self.point: Optional[sublime.Region] = None

    def process(self, usages: bool = False, data: Dict[str, Any] = {}) -> None:
        """
        Process the definitions.
        """
        view: sublime.View = self.text.view
        if not data.get('success', False):
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return

        definitions: Union[str, Definitions] = data['goto'] if not usages else data['usages']
        if isinstance(definitions, list) and len(definitions) == 0:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        if definitions is not None and isinstance(definitions, list) and len(definitions) == 1 and (not usages):
            self._jump(*definitions[0])
        else:
            self._show_options(definitions, usages)

    def _jump(self, filename: Union[int, str], lineno: Optional[int] = None, columno: Optional[int] = None, transient: bool = False) -> None:
        """
        Jump to a window.
        """
        # If filename is an int, check for special case.
        if isinstance(filename, int):
            if filename == -1:
                view: sublime.View = self.text.view
                point: sublime.Region = self.point  # type: ignore
                sublime.active_window().focus_view(view)
                view.show(point)
                if view.sel()[0] != point:
                    view.sel().clear()
                    view.sel().add(point)
                return
        # Lookup options based on filename.
        opts: Any = self.options[filename]  # type: ignore
        if isinstance(self.options, list) and len(self.options) > filename and len(self.options[filename]) == 4:
            opts = opts[1:]
        # Unpack the values.
        filename, lineno, columno = opts  # type: ignore
        flags: int = sublime.ENCODED_POSITION
        if transient:
            flags |= sublime.TRANSIENT
        file_str: str = '{}:{}:{}'.format(filename, lineno or 0, columno or 0)
        sublime.active_window().open_file(file_str, flags)
        self._toggle_indicator(lineno, columno)

    def _show_options(self, defs: Union[str, Definitions], usages: bool) -> None:
        """
        Show a dropdown quickpanel with options to jump.
        """
        view: sublime.View = self.text.view
        options: List[List[str]]
        if usages or (not usages and not isinstance(defs, str)):
            if isinstance(defs, list) and len(defs) > 0 and len(defs[0]) == 4:
                options = [[o[0], o[1], 'line: {} column: {}'.format(o[2], o[3])] for o in defs]  # type: ignore
            elif isinstance(defs, list) and len(defs) > 0:
                options = [[o[0], 'line: {} column: {}'.format(o[1], o[2])] for o in defs]  # type: ignore
            else:
                options = []
        elif isinstance(defs, str):
            options = [defs.split('\n')]
        else:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        self.options = defs
        self.point = view.sel()[0]
        view.window().show_quick_panel(options, self._jump, on_highlight=partial(self._jump, transient=True))

    def _toggle_indicator(self, lineno: int = 0, columno: int = 0) -> None:
        """
        Toggle mark indicator for focusing the cursor.
        """
        pt: int = self.text.view.text_point(lineno - 1, columno)
        region_name: str = 'anaconda.indicator.{}.{}'.format(self.text.view.id(), lineno)
        for i in range(3):
            delta: int = 300 * i * 2
            sublime.set_timeout(lambda: self.text.view.add_regions(region_name, [sublime.Region(pt, pt)], 'comment', 'bookmark', sublime.DRAW_EMPTY_AS_OVERWRITE), delta)
            sublime.set_timeout(lambda: self.text.view.erase_regions(region_name), delta + 300)
