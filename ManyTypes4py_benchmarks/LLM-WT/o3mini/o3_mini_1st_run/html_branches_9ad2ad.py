import re
from collections import defaultdict
from typing import List, DefaultDict, Iterable
from .template_parser import FormattedError, Token, tokenize

class TagInfo:
    def __init__(self, tag: str, classes: List[str], ids: List[str], token: Token) -> None:
        self.tag: str = tag
        self.classes: List[str] = classes
        self.ids: List[str] = ids
        self.token: Token = token
        self.words: List[str] = [self.tag, *('.' + s for s in classes), *('#' + s for s in ids)]

    def text(self) -> str:
        s: str = self.tag
        if self.classes:
            s += '.' + '.'.join(self.classes)
        if self.ids:
            s += '#' + '#'.join(self.ids)
        return s

def get_tag_info(token: Token) -> TagInfo:
    s: str = token.s
    tag: str = token.tag
    classes: List[str] = []
    ids: List[str] = []
    searches: List[tuple[List[str], str]] = [
        (classes, ' class="(.*?)"'),
        (classes, " class='(.*?)'"),
        (ids, ' id="(.*?)"'),
        (ids, " id='(.*?)'")
    ]
    for lst, regex in searches:
        m = re.search(regex, s)
        if m:
            for g in m.groups():
                lst += split_for_id_and_class(g)
    return TagInfo(tag=tag, classes=classes, ids=ids, token=token)

def split_for_id_and_class(element: str) -> List[str]:
    outside_braces: bool = True
    lst: List[str] = []
    s: str = ''
    for ch in element:
        if ch == '{':
            outside_braces = False
        if ch == '}':
            outside_braces = True
        if ch == ' ' and outside_braces:
            if s != '':
                lst.append(s)
            s = ''
        else:
            s += ch
    if s != '':
        lst.append(s)
    return lst

def build_id_dict(templates: Iterable[str]) -> DefaultDict[str, List[str]]:
    template_id_dict: DefaultDict[str, List[str]] = defaultdict(list)
    for fn in templates:
        with open(fn) as f:
            text: str = f.read()
        try:
            list_tags = tokenize(text)
        except FormattedError as e:
            raise Exception(f'\n                fn: {fn}\n                {e}')
        for tag in list_tags:
            info: TagInfo = get_tag_info(tag)
            for id_str in info.ids:
                template_id_dict[id_str].append('Line ' + str(info.token.line) + ':' + fn)
    return template_id_dict