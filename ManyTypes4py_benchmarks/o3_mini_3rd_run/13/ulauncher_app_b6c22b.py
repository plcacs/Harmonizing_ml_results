from __future__ import annotations
import logging
import weakref
from typing import Any, Optional, cast
from gi.repository import Gio, Gtk
import ulauncher
from ulauncher import config
from ulauncher.config import APP_ID, FIRST_RUN
from ulauncher.ui.windows.ulauncher_window import UlauncherWindow
from ulauncher.utils.eventbus import EventBus
from ulauncher.utils.settings import Settings
from ulauncher.utils.singleton import get_instance

logger: logging.Logger = logging.getLogger()
events: EventBus = EventBus('app')

class UlauncherApp(Gtk.Application):
    query: str = ''
    _window: Optional[weakref.ReferenceType[UlauncherWindow]] = None
    _preferences: Optional[weakref.ReferenceType["PreferencesWindow"]] = None
    _tray_icon: Optional[Any] = None  # Replace Any with TrayIcon if imported at module level

    def __call__(self, *args: Any, **kwargs: Any) -> UlauncherApp:
        return cast(UlauncherApp, get_instance(super(), self, *args, **kwargs))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.update(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE)
        super().__init__(*args, **kwargs)
        events.set_self(self)
        self.connect('startup', lambda *_: self.setup())

    @events.on
    def set_query(self, value: str, update_input: bool = True) -> None:
        self.query = value.lstrip()
        if update_input:
            events.emit('window:set_input', self.query)

    def do_startup(self) -> None:
        Gtk.Application.do_startup(self)
        Gio.ActionMap.add_action_entries(
            self,
            [
                ('show-preferences', lambda *_: self.show_preferences(), None),
                ('toggle-tray-icon', lambda *args: self.toggle_tray_icon(args[1].get_boolean()), 'b'),
                ('set-query', lambda *args: self.activate_query(args[1].get_string()), 's')
            ]
        )

    def do_activate(self, *_args: Any, **_kwargs: Any) -> None:
        self.show_launcher()

    def do_command_line(self, command_line: Gio.ApplicationCommandLine, *args: Any, **_kwargs: Any) -> int:
        arguments: list[str] = command_line.get_arguments()
        if '--daemon' not in arguments and '--no-window' not in arguments:
            self.activate()
        return 0

    def setup(self) -> None:
        settings: Settings = Settings.load()
        if not settings.daemonless or config.get_options().daemon:
            self.hold()
            if settings.show_tray_icon:
                self.toggle_tray_icon(True)
        if FIRST_RUN or settings.hotkey_show_app:
            from ulauncher.utils.hotkey_controller import HotkeyController
            if HotkeyController.is_supported():
                hotkey: str = '<Primary>space'
                if settings.hotkey_show_app and (not HotkeyController.is_plasma()):
                    hotkey = settings.hotkey_show_app
                if HotkeyController.setup_default(hotkey):
                    display_name: str = Gtk.accelerator_get_label(*Gtk.accelerator_parse(hotkey))
                    body: str = f'Ulauncher has added a global keyboard shortcut: "{display_name}" to your desktop settings'
                    self.show_notification('de_hotkey_auto_created', 'Global shortcut created', body)
            else:
                body: str = ("Ulauncher doesn't support setting global keyboard shortcuts for your desktop. "
                             "There are more details on this in the preferences view (click here to open).")
                self.show_notification('de_hotkey_unsupported', 'Cannot create global shortcut', body, 'app.show-preferences')
            settings.save(hotkey_show_app='')

    def show_notification(self, notification_id: str, title: str, body: str, default_action: str = '-') -> None:
        notification: Gio.Notification = Gio.Notification.new(title)
        notification.set_default_action(default_action)
        notification.set_body(body)
        notification.set_priority(Gio.NotificationPriority.URGENT)
        self.send_notification(notification_id, notification)

    @events.on
    def show_launcher(self) -> None:
        window: Optional[UlauncherWindow] = self._window() if self._window else None
        if not window:
            self._window = weakref.ref(UlauncherWindow(application=self))

    @events.on
    def show_preferences(self, page: Optional[str] = None) -> None:
        window: Optional[UlauncherWindow] = self._window() if self._window else None
        if window:
            window.close(save_query=True)
        preferences: Optional["PreferencesWindow"] = self._preferences() if self._preferences else None
        if preferences:
            preferences.present(page)
        else:
            from ulauncher.ui.windows.preferences_window import PreferencesWindow
            preferences = PreferencesWindow(application=self)
            self._preferences = weakref.ref(preferences)
            preferences.show(page)

    def activate_query(self, query_str: str) -> None:
        self.activate()
        self.set_query(query_str)

    @events.on
    def toggle_tray_icon(self, enable: bool) -> None:
        if not self._tray_icon:
            from ulauncher.ui.tray_icon import TrayIcon
            self._tray_icon = TrayIcon()
        self._tray_icon.switch(enable)

    @events.on
    def toggle_hold(self, value: bool) -> None:
        if value:
            self.hold()
        else:
            self.release()

    @events.on
    def quit(self) -> None:
        super().quit()