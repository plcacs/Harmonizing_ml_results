import mistune
import schema
import json
import yaml
import os
import copy
import re
import sys
from typing import Dict, List, Any, Set, Tuple, Optional, Union, TextIO, Iterator, cast
import io
import logging
import urllib.parse as urlparse
from aslist import aslist
import argparse

_logger = logging.getLogger("salad")

def has_types(items: Any) -> List[str]:
    r: List[str] = []
    if isinstance(items, dict):
        if items["type"] == "https://w3id.org/cwl/salad#record":
            return [items["name"]]
        for n in ("type", "items", "values"):
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
    return "[%s](#%s)" % (frg, to_id(frg))

class MyRenderer(mistune.Renderer):
    def header(self, text: str, level: int, raw: Optional[str] = None) -> str:
        return """<h%i id="%s">%s</h1>""" % (level, to_id(text), text)

def to_id(text: str) -> str:
    textid = text
    if text[0] in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        try:
            textid = text[text.index(" ")+1:]
        except ValueError:
            pass
    textid = textid.replace(" ", "_")
    return textid

class ToC:
    def __init__(self) -> None:
        self.first_toc_entry: bool = True
        self.numbering: List[int] = [0]
        self.toc: str = ""
        self.start_numbering: bool = True

    def add_entry(self, thisdepth: int, title: str) -> str:
        depth = len(self.numbering)
        if thisdepth < depth:
            self.toc += "</ol>"
            for n in range(0, depth-thisdepth):
                self.numbering.pop()
                self.toc += "</li></ol>"
            self.numbering[-1] += 1
        elif thisdepth == depth:
            if not self.first_toc_entry:
                self.toc += "</ol>"
            else:
                self.first_toc_entry = False
            self.numbering[-1] += 1
        elif thisdepth > depth:
            self.numbering.append(1)

        if self.start_numbering:
            num = "%i.%s" % (self.numbering[0], ".".join([str(n) for n in self.numbering[1:]]))
        else:
            num = ""
        self.toc += """<li><a href="#%s">%s %s</a><ol>\n""" %(to_id(title),
            num, title)
        return num

    def contents(self, id: str) -> str:
        c = """<h1 id="%s">Table of contents</h1>
               <nav class="tocnav"><ol>%s""" % (id, self.toc)
        c += "</ol>"
        for i in range(0, len(self.numbering)):
            c += "</li></ol>"
        c += """</nav>"""
        return c

basicTypes: Tuple[str, ...] = ("https://w3id.org/cwl/salad#null",
              "http://www.w3.org/2001/XMLSchema#boolean",
              "http://www.w3.org/2001/XMLSchema#int",
              "http://www.w3.org/2001/XMLSchema#long",
              "http://www.w3.org/2001/XMLSchema#float",
              "http://www.w3.org/2001/XMLSchema#double",
              "http://www.w3.org/2001/XMLSchema#string",
              "https://w3id.org/cwl/salad#record",
              "https://w3id.org/cwl/salad#enum",
              "https://w3id.org/cwl/salad#array")

def add_dictlist(di: Dict[str, List[str]], key: str, val: str) -> None:
    if key not in di:
        di[key] = []
    di[key].append(val)

def number_headings(toc: ToC, maindoc: str) -> str:
    mdlines: List[str] = []
    skip = False
    for line in maindoc.splitlines():
        if line.strip() == "# Introduction":
            toc.start_numbering = True
            toc.numbering = [0]

        if line == "