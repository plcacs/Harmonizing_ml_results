"""Common jedi class to work with Jedi functions
"""
import sublime
from functools import partial
from typing import Union, Dict, Optional, Any, List, Tuple


class JediUsages:
    """Work with Jedi definitions
    """

    def __init__(self, text: Any) -> None:
        self.text = text

    def process(self, usages: bool = False, data: Optional[Dict[str, Any]] = None) -> None:
        """Process the definitions
        """
        view: sublime.View = self.text.view
        if not data or not data.get('success', False):
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        definitions: Optional[List[Tuple[Union[str, int], Optional[int], Optional[int], Optional[Any]]]] = data['goto'] if not usages else data['usages']
        if not definitions:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        if definitions and len(definitions) == 1 and not usages:
            self._jump(*definitions[0])
        else:
            self._show_options(definitions, usages)

    def _jump(
        self,
        filename: Union[str, int],
        lineno: Optional[int] = None,
        columno: Optional[int] = None,
        transient: bool = False
    ) -> None:
        """Jump to a window
        """
        if isinstance(filename, int):
            if filename == -1:
                view: sublime.View = self.text.view
                point: int = self.point
                sublime.active_window().focus_view(view)
                view.show(point)
                if view.sel()[0] != point:
                    view.sel().clear()
                    view.sel().add(point)
                return
        opts: Tuple[Any, ...] = self.options[filename]
        if len(self.options[filename]) == 4:
            opts = opts[1:]
        filename, lineno, columno = opts
        flags: int = sublime.ENCODED_POSITION
        if transient:
            flags |= sublime.TRANSIENT
        sublime.active_window().open_file(f'{filename}:{lineno or 0}:{columno or 0}', flags)
        self._toggle_indicator(lineno, columno)

    def _show_options(
        self,
        defs: Union[List[Tuple[Union[str, int], Optional[int], Optional[int], Optional[Any]]], str],
        usages: bool
    ) -> None:
        """Show a dropdown quickpanel with options to jump
        """
        view: sublime.View = self.text.view
        if usages or (not usages and not isinstance(defs, str)):
            if len(defs) == 4:
                options: List[List[str]] = [
                    [str(o[0]), str(o[1]), f'line: {o[2]} column: {o[3]}'] for o in defs
                ]
            else:
                options = [
                    [str(o[0]), f'line: {o[1]} column: {o[2]}'] for o in defs
                ]
        elif len(defs):
            options = defs[0]  # type: ignore
        else:
            sublime.status_message('Unable to find {}'.format(view.substr(view.word(view.sel()[0]))))
            return
        self.options: Union[List[Tuple[Union[str, int], Optional[int], Optional[int], Optional[Any]]], str] = defs
        self.point: int = self.text.view.sel()[0].begin()
        self.text.view.window().show_quick_panel(
            options,
            self._jump,
            on_highlight=partial(self._jump, transient=True)
        )

    def _toggle_indicator(self, lineno: int = 0, columno: int = 0) -> None:
        """Toggle mark indicator for focus the cursor
        """
        pt: int = self.text.view.text_point(lineno - 1, columno)
        region_name: str = f'anaconda.indicator.{self.text.view.id()}.{lineno}'
        for i in range(3):
            delta: int = 300 * i * 2
            sublime.set_timeout(
                lambda rn=region_name, p=pt: self.text.view.add_regions(
                    rn,
                    [sublime.Region(p, p)],
                    'comment',
                    'bookmark',
                    sublime.DRAW_EMPTY_AS_OVERWRITE
                ),
                delta
            )
            sublime.set_timeout(
                lambda rn=region_name: self.text.view.erase_regions(rn),
                delta + 300
            )
