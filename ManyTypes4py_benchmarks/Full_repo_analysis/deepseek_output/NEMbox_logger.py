import logging
from typing import cast
from . import const
FILE_NAME: str = const.Constant.log_path
with open(FILE_NAME, 'a+') as f:
    f.write('#' * 80)
    f.write('\n')

def getLogger(name: str) -> logging.Logger:
    log: logging.Logger = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    fh: logging.FileHandler = logging.FileHandler(FILE_NAME)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(lineno)s: %(message)s'))
    log.addHandler(fh)
    return log
