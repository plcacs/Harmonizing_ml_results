import sys
from .main import main

fixes: str = 'lib2to3.fixes'
exit_code: int = main(fixes)
sys.exit(exit_code)
