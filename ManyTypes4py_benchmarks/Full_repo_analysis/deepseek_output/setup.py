import os
import re
from typing import List, Dict, Any, Optional
from setuptools import setup

def get_version(package: str) -> str:
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with open(os.path.join(package, '__init__.py')) as f:
        match = re.search('__version__ = [\'"]([^\'"]+)[\'"]', f.read())
        if match is None:
            raise ValueError("Could not find __version__ in package init")
        return match.group(1)

def get_long_description() -> str:
    """
    Return the README.
    """
    with open('README.md', encoding='utf8') as f:
        return f.read()

def get_packages(package: str) -> List[str]:
    """
    Return root package and all sub-packages.
    """
    return [dirpath for (dirpath, dirnames, filenames) in os.walk(package) if os.path.exists(os.path.join(dirpath, '__init__.py'))]

setup(
    name='databases',
    version=get_version('databases'),
    python_requires='>=3.8',
    url='https://github.com/encode/databases',
    license='BSD',
    description='Async database support for Python.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Tom Christie',
    author_email='tom@tomchristie.com',
    packages=get_packages('databases'),
    package_data={'databases': ['py.typed']},
    install_requires=['sqlalchemy>=2.0.7'],
    extras_require={
        'postgresql': ['asyncpg'],
        'asyncpg': ['asyncpg'],
        'aiopg': ['aiopg'],
        'mysql': ['aiomysql'],
        'aiomysql': ['aiomysql'],
        'asyncmy': ['asyncmy'],
        'sqlite': ['aiosqlite'],
        'aiosqlite': ['aiosqlite']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Internet :: WWW/HTTP',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3 :: Only'
    ],
    zip_safe=False
)
