"""
Iterate on commits of the asyncio Git repository using the Dulwich module.
"""
import os
import pyperf
from dulwich.repo import Repo, Walker
from typing import Any

def iter_all_commits(repo: Repo) -> None:
    head: bytes = repo.head()
    walker: Walker = repo.get_walker(head)
    for entry in walker:
        pass

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Dulwich benchmark: iterate on all Git commits'
    repo_path: str = os.path.join(os.path.dirname(__file__), 'data', 'asyncio.git')
    repo: Repo = Repo(repo_path)
    runner.bench_func('dulwich_log', iter_all_commits, repo)
    repo.close()
