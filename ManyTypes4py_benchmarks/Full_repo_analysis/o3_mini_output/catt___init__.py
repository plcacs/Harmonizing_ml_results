import sys
from typing import NoReturn

if sys.version_info.major < 3:
    print('This program requires Python 3 and above to run.')
    sys.exit(1)  # type: NoReturn

__codename__: str = 'Zaniest Zapper'