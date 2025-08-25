import sys
from .main import main

exit_code: int = main('lib2to3.fixes')
sys.exit(exit_code)
