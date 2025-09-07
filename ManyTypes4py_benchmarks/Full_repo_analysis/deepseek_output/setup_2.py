import io
import os
import sys
from shutil import rmtree
from setuptools import setup, Command
from typing import List, Dict, Any, Optional, IO

NAME: str = 'requests-html'
DESCRIPTION: str = 'HTML Parsing for Humans.'
URL: str = 'https://github.com/psf/requests-html'
EMAIL: str = 'me@kennethreitz.org'
AUTHOR: str = 'Kenneth Reitz'
VERSION: str = '0.10.0'
REQUIRED: List[str] = ['requests', 'pyquery', 'fake-useragent', 'parse', 'beautifulsoup4', 'w3lib', 'pyppeteer>=0.0.14']
here: str = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description: str = '\n' + f.read()

class UploadCommand(Command):
    """Support setup.py upload."""
    description: str = 'Build and publish the package.'
    user_options: List[Any] = []

    @staticmethod
    def status(s: str) -> None:
        """Prints things in bold."""
        print('\x1b[1m{0}\x1b[0m'.format(s))

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass
        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))
        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')
        self.status('Publishing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')
        sys.exit()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    python_requires='>=3.6.0',
    py_modules=['requests_html'],
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    cmdclass={'upload': UploadCommand}
)
