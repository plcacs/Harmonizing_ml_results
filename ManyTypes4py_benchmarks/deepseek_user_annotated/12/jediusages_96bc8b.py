# Copyright (C) 2013 - Oscar Campos <oscar.campos@member.fsf.org>
# This program is Free Software see LICENSE file for details

"""Common jedi class to work with Jedi functions
"""

import sublime
from functools import partial
from typing import List, Tuple, Any, Optional

from ._typing import Union, Dict


class JediUsages(object):
    """Work with Jedi definitions
    """

    def __init__(self, text: Any) -> None:
        self.text = text
        self.options: Dict[Union[int, str], Tuple[Any, ...]] = {}
        self.point: sublime.Region = sublime.Region(0)

    def process(self, usages: bool = False, data: Optional[Dict[str, Any]] = None) -> None:
        """Process the definitions
        """

        view = self.text.view
        if not data or not data['success']:
            sublime.status_message('Unable to find {}'.format(
                view.substr(view.word(view.sel()[0])))
            return

        definitions = data['goto'] if not usages else data['usages']
        if len(definitions) == 0:
            sublime.status_message('Unable to find {}'.format(
                view.substr(view.word(view.sel()[0])))
            return

        if definitions is not None and len(definitions) == 1 and not usages:
            self._jump(*definitions[0])
        else:
            self._show_options(definitions, usages)

    def _jump(self, filename: Union[int, str], lineno: Optional[int] = None,
              columno: Optional[int] = None, transient: bool = False) -> None:
        """Jump to a window
        """

        # process jumps from options window
        if type(filename) is int:
            if filename == -1:
                # restore view
                view = self.text.view
                point = self.point

                sublime.active_window().focus_view(view)
                view.show(point)

                if view.sel()[0] != point:
                    view.sel().clear()
                    view.sel().add(point)

                return

        opts = self.options[filename]
        if len(self.options[filename]) == 4:
            opts = opts[1:]

        filename, lineno, columno = opts  # type: ignore
        flags = sublime.ENCODED_POSITION
        if transient:
            flags |= sublime.TRANSIENT

        sublime.active_window().open_file(
            '{}:{}:{}'.format(filename, lineno or 0, columno or 0),
            flags
        )

        self._toggle_indicator(lineno or 0, columno or 0)

    def _show_options(self, defs: Union[str, List[Tuple[Any, ...]]], usages: bool) -> None:
        """Show a dropdown quickpanel with options to jump
        """

        view = self.text.view
        if usages or (not usages and type(defs) is not str):
            if len(defs[0]) == 4:  # type: ignore
                options = [[
                    o[0], o[1], 'line: {} column: {}'.format(o[2], o[3])
                ] for o in defs]  # type: ignore
            else:
                options = [[
                    o[0], 'line: {} column: {}'.format(o[1], o[2])
                ] for o in defs]  # type: ignore
        else:
            if len(defs):  # type: ignore
                options = defs[0]  # type: ignore
            else:
                sublime.status_message('Unable to find {}'.format(
                    view.substr(view.word(view.sel()[0])))
                return

        self.options = {i: opt for i, opt in enumerate(defs)}  # type: ignore
        self.point = self.text.view.sel()[0]
        self.text.view.window().show_quick_panel(
            options, self._jump,
            on_highlight=partial(self._jump, transient=True)
        )

    def _toggle_indicator(self, lineno: int = 0, columno: int = 0) -> None:
        """Toggle mark indicator for focus the cursor
        """

        pt = self.text.view.text_point(lineno - 1, columno)
        region_name = 'anaconda.indicator.{}.{}'.format(
            self.text.view.id(), lineno
        )

        for i in range(3):
            delta = 300 * i * 2
            sublime.set_timeout(lambda: self.text.view.add_regions(
                region_name,
                [sublime.Region(pt, pt)],
                'comment',
                'bookmark',
                sublime.DRAW_EMPTY_AS_OVERWRITE
            ), delta)
            sublime.set_timeout(
                lambda: self.text.view.erase_regions(region_name),
                delta + 300
            )
