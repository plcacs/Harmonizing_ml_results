import os
import glob
import logging
from string import Template
import sublime
from .helpers import get_settings
from ._typing import Callable, Union, Dict

class Tooltip(object):
    themes: Dict[str, str] = {}
    tooltips: Dict[str, Template] = {}
    loaded: bool = False
    basesize: int = 75

    def __init__(self, theme: str) -> None:
        self.theme: str = theme
        if int(sublime.version()) < 3070:
            return
        if Tooltip.loaded is False:
            self._load_css_themes()
            self._load_tooltips()
            Tooltip.loaded = True

    def show_tooltip(self, view, tooltip, content, fallback: Callable[[], None]) -> None:
        st_ver: int = int(sublime.version())
        if st_ver < 3070:
            return fallback()
        width: int = get_settings(view, 'font_size', 8) * 75
        kwargs: Dict[str, Union[int, str]] = {'location': -1, 'max_width': width if width < 900 else 900}
        if st_ver >= 3071:
            kwargs['flags'] = sublime.COOPERATE_WITH_AUTO_COMPLETE
        text: str = self._generate(tooltip, content)
        if text is None:
            return fallback()
        return view.show_popup(text, **kwargs)

    def _generate(self, tooltip: str, content: Dict[str, str]) -> Union[str, None]:
        try:
            t: str = self.theme
            theme: str = self.themes[t] if t in self.themes else self.themes['popup']
            context: Dict[str, str] = {'css': theme}
            context.update(content)
            data: str = self.tooltips[tooltip].safe_substitute(context)
            return data
        except KeyError as err:
            logging.error(f"while generating tooltip: tooltip {str(err)} don't exists")
            return None

    def _load_tooltips(self) -> None:
        template_files_pattern: str = os.path.join(os.path.dirname(__file__), os.pardir, 'templates', 'tooltips', '*.tpl')
        for template_file in glob.glob(template_files_pattern):
            with open(template_file, 'r', encoding='utf8') as tplfile:
                tplname: str = os.path.basename(template_file).split('.tpl')[0]
                tpldata: str = '<style>${{css}}</style>{}'.format(tplfile.read())
                self.tooltips[tplname] = Template(tpldata)

    def _load_css_themes(self) -> None:
        css_files_pattern: str = os.path.join(os.path.dirname(__file__), os.pardir, 'css', '*.css')
        for css_file in glob.glob(css_files_pattern):
            logging.info(f'anaconda: {self._load_css(css_file)} css theme loaded')
        packages: str = sublime.active_window().extract_variables()['packages']
        user_css_path: str = os.path.join(packages, 'User', 'Anaconda.themes')
        if os.path.exists(user_css_path):
            css_files_pattern = os.path.join(user_css_path, '*.css')
            for css_file in glob.glob(css_files_pattern):
                logging.info(f'anaconda: {self._load_css(css_file)} user css theme loaded')

    def _load_css(self, css_file: str) -> str:
        theme_name: str = os.path.basename(css_file).split('.css')[0]
        with open(css_file, 'r') as resource:
            self.themes[theme_name] = resource.read()
        return theme_name
