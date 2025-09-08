from os import path
from setuptools import setup
from typing import Dict, List, Any, Optional

this_directory: str = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description: str = f.read()

setup(
    name: str = 'easyquotation',
    version: str = '0.7.5',
    description: str = 'A utility for Fetch China Stock Info',
    long_description: str = long_description,
    long_description_content_type: str = 'text/markdown',
    author: str = 'shidenggui',
    author_email: str = 'longlyshidenggui@gmail.com',
    license: str = 'BSD',
    url: str = 'https://github.com/shidenggui/easyquotation',
    keywords: str = 'China stock trade',
    install_requires: List[str] = ['requests', 'six', 'easyutils'],
    classifiers: List[str] = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: BSD License'
    ],
    packages: List[str] = ['easyquotation'],
    package_data: Dict[str, List[str]] = {'': ['*.conf']}
)
