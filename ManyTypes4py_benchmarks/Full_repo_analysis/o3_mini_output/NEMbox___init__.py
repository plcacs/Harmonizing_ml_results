from importlib_metadata import version
from .const import Constant
from .utils import create_dir
from .utils import create_file

__version__: str = version('NetEase-MusicBox')

def create_config() -> None:
    create_dir(Constant.conf_dir)
    create_dir(Constant.download_dir)
    create_file(Constant.storage_path)
    create_file(Constant.log_path, default='')
    create_file(Constant.cookie_path, default='# Netscape HTTP Cookie File\n')

create_config()