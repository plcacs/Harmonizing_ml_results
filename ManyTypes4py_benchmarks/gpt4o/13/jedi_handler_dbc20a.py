import logging
import jedi
from lib.anaconda_handler import AnacondaHandler
from jedi.api import refactoring as jedi_refactor
from commands import Doc, Goto, GotoAssignment, Rename, FindUsages
from commands import CompleteParameters, AutoComplete
from typing import Any, Dict

logger = logging.getLogger('')

class JediHandler(AnacondaHandler):
    """Handle requests to execute Jedi related commands to the JsonServer

    The JsonServer instantiate an object of this class passing the method
    to execute as it came from the Sublime Text 3 Anaconda plugin
    """

    def run(self) -> None:
        """Call the specific method (override base class)"""
        self.real_callback = self.callback
        self.callback = self.handle_result_and_purge_cache
        super(JediHandler, self).run()

    def handle_result_and_purge_cache(self, result: Any) -> None:
        """Handle the result from the call and purge in memory jedi cache"""
        try:
            jedi.cache.clear_time_caches()
        except:
            jedi.cache.clear_caches()
        self.real_callback(result)

    @property
    def script(self) -> jedi.Script:
        """Generates a new valid Jedi Script and return it back"""
        return self.jedi_script(**self.data)

    def jedi_script(self, source: str, line: int, offset: int, filename: str = '', encoding: str = 'utf8', **kw: Any) -> jedi.Script:
        """Generate an usable Jedi Script"""
        jedi_project = jedi.get_default_project(filename)
        return jedi.Script(source, project=jedi_project)

    def rename(self, directories: Any, new_word: str) -> None:
        """Rename the object under the cursor by the given word"""
        Rename(self.callback, self.uid, self.script, directories, new_word, jedi_refactor)

    def autocomplete(self) -> None:
        """Call autocomplete"""
        AutoComplete(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def parameters(self) -> None:
        """Call complete parameter"""
        CompleteParameters(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script, self.settings)

    def usages(self) -> None:
        """Call find usages"""
        FindUsages(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def goto(self) -> None:
        """Call goto"""
        Goto(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def goto_assignment(self) -> None:
        """Call goto_assignment"""
        GotoAssignment(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script)

    def doc(self, html: bool = False) -> None:
        """Call doc"""
        Doc(self.callback, self.data.get('line', 1), self.data.get('offset', 0), self.uid, self.script, html)
