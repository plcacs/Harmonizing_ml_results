import logging
from typing import Any, Dict, List, Union
import jedi
from lib.anaconda_handler import AnacondaHandler
from jedi.api import refactoring as jedi_refactor
from commands import Doc, Goto, GotoAssignment, Rename, FindUsages
from commands import CompleteParameters, AutoComplete

logger: logging.Logger = logging.getLogger('')

class JediHandler(AnacondaHandler):
    def run(self) -> None:
        self.real_callback: Any = self.callback
        self.callback = self.handle_result_and_purge_cache
        super(JediHandler, self).run()

    def handle_result_and_purge_cache(self, result: Any) -> None:
        try:
            jedi.cache.clear_time_caches()
        except:
            jedi.cache.clear_caches()
        self.real_callback(result)

    @property
    def script(self) -> Any:
        return self.jedi_script(**self.data)

    def jedi_script(self, source: str, line: int, offset: int, filename: str = '', encoding: str = 'utf8', **kw: Any) -> Any:
        jedi_project = jedi.get_default_project(filename)
        return jedi.Script(source, project=jedi_project)

    def rename(self, directories: List[str], new_word: str) -> None:
        Rename(self.callback, self.uid, self.script, directories, new_word, jedi_refactor)

    def autocomplete(self) -> None:
        AutoComplete(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def parameters(self) -> None:
        CompleteParameters(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script, self.settings)

    def usages(self) -> None:
        FindUsages(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def goto(self) -> None:
        Goto(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def goto_assignment(self) -> None:
        GotoAssignment(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def doc(self, html: bool = False) -> None:
        Doc(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script, html)
