import os
from typing import Final

class Constant(object):
    if 'XDG_CONFIG_HOME' in os.environ:
        conf_dir: Final[str] = os.path.join(os.environ['XDG_CONFIG_HOME'], 'netease-musicbox')
    else:
        conf_dir: Final[str] = os.path.join(os.path.expanduser('~'), '.netease-musicbox')
    config_path: Final[str] = os.path.join(conf_dir, 'config.json')

    if 'XDG_CACHE_HOME' in os.environ:
        cacheDir: Final[str] = os.path.join(os.environ['XDG_CACHE_HOME'], 'netease-musicbox')
        if not os.path.exists(cacheDir):
            os.mkdir(cacheDir)
        download_dir: Final[str] = os.path.join(cacheDir, 'cached')
        cache_path: Final[str] = os.path.join(cacheDir, 'nemcache')
    else:
        download_dir: Final[str] = os.path.join(conf_dir, 'cached')
        cache_path: Final[str] = os.path.join(conf_dir, 'nemcache')

    if 'XDG_DATA_HOME' in os.environ:
        dataDir: Final[str] = os.path.join(os.environ['XDG_DATA_HOME'], 'netease-musicbox')
        if not os.path.exists(dataDir):
            os.mkdir(dataDir)
        cookie_path: Final[str] = os.path.join(dataDir, 'cookie.txt')
        log_path: Final[str] = os.path.join(dataDir, 'musicbox.log')
        storage_path: Final[str] = os.path.join(dataDir, 'database.json')
    else:
        cookie_path: Final[str] = os.path.join(conf_dir, 'cookie.txt')
        log_path: Final[str] = os.path.join(conf_dir, 'musicbox.log')
        storage_path: Final[str] = os.path.join(conf_dir, 'database.json')