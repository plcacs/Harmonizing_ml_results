import mistune
import schema
import json
import yaml
import os
import copy
import re
import sys
import logging
import urlparse
from aslist import aslist
import argparse
from typing import Any, List, Dict, Set, Tuple, Union, Optional
from io import StringIO

_logger = logging.getLogger('salad')


def has_types(items: Any) -> List[str]:
    r: List[str] = []
    if isinstance(items, dict):
        if items.get('type') == 'https://w3id.org/cwl/salad#record':
            return [items.get('name', '')]
        for n in ('type', 'items', 'values'):
            if n in items:
                r.extend(has_types(items[n]))
        return r
    if isinstance(items, list):
        for i in items:
            r.extend(has_types(i))
        return r
    if isinstance(items, str):
        return [items]
    return []


def linkto(item: str) -> str:
    _, frg = urlparse.urldefrag(item)
    return f'[{frg}](#{to_id(frg)})'


class MyRenderer(mistune.Renderer):
    def header(self, text: str, level: int, raw: Optional[str] = None) -> str:
        return f'<h{level} id="{to_id(text)}">{text}</h{level}>'


def to_id(text: str) -> str:
    textid = text
    if text and text[0].isdigit():
        try:
            textid = text[text.index(' ') + 1:]
        except ValueError:
            pass
    textid = textid.replace(' ', '_')
    return textid


class ToC:
    def __init__(self) -> None:
        self.first_toc_entry: bool = True
        self.numbering: List[int] = [0]
        self.toc: str = ''
        self.start_numbering: bool = True

    def add_entry(self, thisdepth: int, title: str) -> str:
        depth = len(self.numbering)
        if thisdepth < depth:
            self.toc += '</ol>'
            for _ in range(0, depth - thisdepth):
                self.numbering.pop()
                self.toc += '</li></ol>'
            self.numbering[-1] += 1
        elif thisdepth == depth:
            if not self.first_toc_entry:
                self.toc += '</ol>'
            else:
                self.first_toc_entry = False
            self.numbering[-1] += 1
        elif thisdepth > depth:
            self.numbering.append(1)
        if self.start_numbering:
            num = f"{self.numbering[0]}." + ".".join(map(str, self.numbering[1:]))
        else:
            num = ''
        self.toc += f'<li><a href="#{to_id(title)}">{num} {title}</a><ol>\n'
        return num

    def contents(self, id: str) -> str:
        c = f'<h1 id="{id}">Table of contents</h1>\n               <nav class="tocnav"><ol>{self.toc}'
        c += '</ol>'
        for _ in range(0, len(self.numbering)):
            c += '</li></ol>'
        c += '</nav>'
        return c


basicTypes: Tuple[str, ...] = (
    'https://w3id.org/cwl/salad#null',
    'http://www.w3.org/2001/XMLSchema#boolean',
    'http://www.w3.org/2001/XMLSchema#int',
    'http://www.w3.org/2001/XMLSchema#long',
    'http://www.w3.org/2001/XMLSchema#float',
    'http://www.w3.org/2001/XMLSchema#double',
    'http://www.w3.org/2001/XMLSchema#string',
    'https://w3id.org/cwl/salad#record',
    'https://w3id.org/cwl/salad#enum',
    'https://w3id.org/cwl/salad#array'
)


def add_dictlist(di: Dict[str, List[str]], key: str, val: str) -> None:
    if key not in di:
        di[key] = []
    di[key].append(val)


def number_headings(toc: ToC, maindoc: str) -> str:
    mdlines: List[str] = []
    skip: bool = False
    for line in maindoc.splitlines():
        if line.strip() == '# Introduction':
            toc.start_numbering = True
            toc.numbering = [0]
        if line == '