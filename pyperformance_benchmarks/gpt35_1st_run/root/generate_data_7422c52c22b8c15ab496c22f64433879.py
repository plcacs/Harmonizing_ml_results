from urllib.request import urlopen
import json
import toml
from typing import List, Dict

BASE_URL: str = 'https://api.github.com/repos/python/cpython/pulls?per_page=1000&state=all'

def main() -> None:
    all_issues: List[Dict[str, str]] = []
    for page in range(1, 11):
        with urlopen(f'{BASE_URL}&page={page}') as response:
            issues: List[Dict[str, str]] = json.loads(response.read())
            if (not issues):
                break
            all_issues.extend(issues)
            print(f'Page: {page} Total Issues: {len(all_issues)}')
    with open('issues.toml', 'w') as f:
        f.write(toml.dumps({'data': all_issues}))
if (__name__ == '__main__'):
    main()
