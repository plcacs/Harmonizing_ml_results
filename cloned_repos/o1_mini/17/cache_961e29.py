"""
Class to cache songs into local storage.
"""
import os
import signal
import subprocess
import threading
from typing import Callable, List, Optional
from . import logger
from .api import NetEase
from .config import Config
from .const import Constant
from .singleton import Singleton

log = logger.getLogger(__name__)

class Cache(Singleton):

    const: Constant
    config: Config
    download_lock: threading.Lock
    check_lock: threading.Lock
    downloading: List[List[object]]
    aria2c: Optional[subprocess.Popen]
    wget: Optional[subprocess.Popen]
    stop: bool
    enable: bool
    aria2c_parameters: List[str]

    def __init__(self) -> None:
        if hasattr(self, '_init'):
            return
        self._init = True
        self.const = Constant()
        self.config = Config()
        self.download_lock = threading.Lock()
        self.check_lock = threading.Lock()
        self.downloading = []
        self.aria2c = None
        self.wget = None
        self.stop = False
        self.enable = self.config.get('cache')
        self.aria2c_parameters = self.config.get('aria2c_parameters')

    def _is_cache_successful(self) -> bool:

        def succ(x: Optional[subprocess.Popen]) -> bool:
            return x is not None and x.returncode == 0
        return succ(self.aria2c) or succ(self.wget)

    def _kill_all(self) -> None:

        def _kill(p: Optional[subprocess.Popen]) -> None:
            if p:
                os.kill(p.pid, signal.SIGKILL)
        _kill(self.aria2c)
        _kill(self.wget)

    def start_download(self) -> Optional[bool]:
        check: bool = self.download_lock.acquire(False)
        if not check:
            return False
        while True:
            if self.stop:
                break
            if not self.enable:
                break
            self.check_lock.acquire()
            if len(self.downloading) <= 0:
                self.check_lock.release()
                break
            data = self.downloading.pop()
            self.check_lock.release()
            song_id: str = data[0]
            song_name: str = data[1]
            artist: str = data[2]
            url: str = data[3]
            onExit: Callable[[str, str], None] = data[4]
            output_path: str = Constant.download_dir
            output_file: str = f"{artist} - {song_name}.mp3"
            full_path: str = os.path.join(output_path, output_file)
            new_url: Optional[str] = NetEase().songs_url([song_id])[0].get('url')
            if new_url:
                log.info(f'Old:{url}. New:{new_url}')
                try:
                    para: List[str] = ['aria2c', '--auto-file-renaming=false', '--allow-overwrite=true', '-d', output_path, '-o', output_file, new_url]
                    para.extend(self.aria2c_parameters)
                    log.debug(para)
                    self.aria2c = subprocess.Popen(para, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.aria2c.wait()
                except OSError as e:
                    log.warning(f'{e}.\tAria2c is unavailable, fall back to wget')
                    para = ['wget', '-O', full_path, new_url]
                    self.wget = subprocess.Popen(para, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.wget.wait()
                if self._is_cache_successful():
                    log.debug(f"{song_id} Cache OK")
                    onExit(song_id, full_path)
        self.download_lock.release()

    def add(self, song_id: str, song_name: str, artist: str, url: str, onExit: Callable[[str, str], None]) -> None:
        self.check_lock.acquire()
        self.downloading.append([song_id, song_name, artist, url, onExit])
        self.check_lock.release()

    def quit(self) -> None:
        self.stop = True
        try:
            self._kill_all()
        except (AttributeError, OSError) as e:
            log.error(e)
            pass
