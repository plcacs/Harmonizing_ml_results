"""
Iterate on commits of the asyncio Git repository using the Dulwich module.
"""
import os
from typing import Iterator, Any
import pyperf
import dulwich.repo

def iter_all_commits(repo: dulwich.repo.Repo) -> Iterator[Any]:
    for entry in repo.get_walker(head):
        pass

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Dulwich benchmark: iterate on all Git commits'
    repo_path = os.path.join(os.path.dirname(__file__), 'data', 'asyncio.git')
    repo = dulwich.repo.Repo(repo_path)
    head = repo.head()
    runner.bench_func('dulwich_log', iter_all_commits, repo)
    repo.close()
