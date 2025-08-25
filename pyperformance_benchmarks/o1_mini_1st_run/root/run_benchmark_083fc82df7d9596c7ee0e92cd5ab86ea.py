"""
Iterate on commits of the asyncio Git repository using the Dulwich module.
"""
import os
from typing import Iterator
import pyperf
from dulwich.repo import Repo
from dulwich.walk import Walker


def iter_all_commits(repo: Repo) -> None:
    for entry in repo.get_walker(head):
        pass


if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Dulwich benchmark: iterate on all Git commits'
    repo_path: str = os.path.join(os.path.dirname(__file__), 'data', 'asyncio.git')
    repo: Repo = Repo(repo_path)
    head: bytes = repo.head()
    runner.bench_func('dulwich_log', iter_all_commits, repo)
    repo.close()
