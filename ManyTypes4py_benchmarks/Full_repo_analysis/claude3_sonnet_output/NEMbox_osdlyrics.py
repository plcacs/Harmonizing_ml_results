import sys
from multiprocessing import Process
from multiprocessing import set_start_method
from typing import Optional, Tuple, List, Any, Union, cast
from . import logger
from .config import Config

log = logger.getLogger(__name__)
config = Config()

try:
    from qtpy import QtGui, QtCore, QtWidgets
    import dbus
    import dbus.service
    import dbus.mainloop.glib
    pyqt_activity: bool = True
except ImportError:
    pyqt_activity: bool = False
    log.warn('qtpy module not installed.')
    log.warn('Osdlyrics Not Available.')

if pyqt_activity:
    QWidget = QtWidgets.QWidget
    QApplication = QtWidgets.QApplication

    class Lyrics(QWidget):
        mpos: QtCore.QPoint

        def __init__(self) -> None:
            super(Lyrics, self).__init__()
            self.text: str = ''
            self.initUI()

        def initUI(self) -> None:
            self.setStyleSheet('background:' + config.get('osdlyrics_background'))
            if config.get('osdlyrics_transparent'):
                self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
            self.setAttribute(QtCore.Qt.WA_X11DoNotAcceptFocus)
            self.setFocusPolicy(QtCore.Qt.NoFocus)
            if config.get('osdlyrics_on_top'):
                self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.X11BypassWindowManagerHint)
            else:
                self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.setMinimumSize(600, 50)
            osdlyrics_size: List[int] = config.get('osdlyrics_size')
            self.resize(osdlyrics_size[0], osdlyrics_size[1])
            scn: int = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
            bl: QtCore.QPoint = QApplication.desktop().screenGeometry(scn).bottomLeft()
            br: QtCore.QPoint = QApplication.desktop().screenGeometry(scn).bottomRight()
            bc: QtCore.QPoint = (bl + br) / 2
            frameGeo: QtCore.QRect = self.frameGeometry()
            frameGeo.moveCenter(bc)
            frameGeo.moveBottom(bc.y())
            self.move(frameGeo.topLeft())
            self.text = 'OSD Lyrics for Musicbox'
            self.setWindowTitle('Lyrics')
            self.show()

        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
            self.mpos = event.pos()

        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
            if event.buttons() and QtCore.Qt.LeftButton:
                diff: QtCore.QPoint = event.pos() - self.mpos
                newpos: QtCore.QPoint = self.pos() + diff
                self.move(newpos)

        def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
            self.resize(self.width() + event.delta(), self.height())

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:
            qp: QtGui.QPainter = QtGui.QPainter()
            qp.begin(self)
            self.drawText(event, qp)
            qp.end()

        def drawText(self, event: QtGui.QPaintEvent, qp: QtGui.QPainter) -> None:
            osdlyrics_color: List[int] = config.get('osdlyrics_color')
            osdlyrics_font: List[Union[str, int]] = config.get('osdlyrics_font')
            font: QtGui.QFont = QtGui.QFont(cast(str, osdlyrics_font[0]), cast(int, osdlyrics_font[1]))
            pen: QtGui.QColor = QtGui.QColor(osdlyrics_color[0], osdlyrics_color[1], osdlyrics_color[2])
            qp.setFont(font)
            qp.setPen(pen)
            qp.drawText(event.rect(), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.text)

        def setText(self, text: str) -> None:
            self.text = text
            self.repaint()

    class LyricsAdapter(dbus.service.Object):
        def __init__(self, name: dbus.service.BusName, session: dbus.SessionBus) -> None:
            dbus.service.Object.__init__(self, name, session)
            self.widget: Lyrics = Lyrics()

        @dbus.service.method('local.musicbox.Lyrics', in_signature='s', out_signature='')
        def refresh_lyrics(self, text: str) -> None:
            self.widget.setText(text.replace('||', '\n'))

        @dbus.service.method('local.musicbox.Lyrics', in_signature='', out_signature='')
        def exit(self) -> None:
            QApplication.quit()

    def show_lyrics() -> None:
        app: QApplication = QApplication(sys.argv)
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        session_bus: dbus.SessionBus = dbus.SessionBus()
        name: dbus.service.BusName = dbus.service.BusName('org.musicbox.Bus', session_bus)
        lyrics: LyricsAdapter = LyricsAdapter(session_bus, '/')
        app.exec_()

def stop_lyrics_process() -> None:
    if pyqt_activity:
        bus: dbus.proxies.ProxyObject = dbus.SessionBus().get_object('org.musicbox.Bus', '/')
        bus.exit(dbus_interface='local.musicbox.Lyrics')

def show_lyrics_new_process() -> None:
    if pyqt_activity and config.get('osdlyrics'):
        set_start_method('spawn')
        p: Process = Process(target=show_lyrics)
        p.daemon = True
        p.start()
