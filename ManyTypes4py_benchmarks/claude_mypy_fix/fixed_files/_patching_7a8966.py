"""
Write patches which add @example() decorators for discovered test cases.

Requires `hypothesis[codemods,ghostwriter]` installed, i.e. black and libcst.

This module is used by Hypothesis' builtin pytest plugin for failing examples
discovered during testing, and by HypoFuzz for _covering_ examples discovered
during fuzzing.
"""
import ast
import difflib
import hashlib
import inspect
import re
import sys
import types
from ast import literal_eval
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import libcst as cst
from libcst import matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.metadata import ExpressionContext, ExpressionContextProvider
from hypothesis.configuration import storage_directory
from hypothesis.version import __version__

try:
    import black
except ImportError:
    black = None

HEADER = f'From HEAD Mon Sep 17 00:00:00 2001\nFrom: Hypothesis {__version__} <no-reply@hypothesis.works>\nDate: {{when:%a, %d %b %Y %H:%M:%S}}\nSubject: [PATCH] {{msg}}\n\n---\n'
FAIL_MSG = 'discovered failure'
_space_only_re = re.compile('^ +$', re.MULTILINE)
_leading_space_re = re.compile('(^[ ]*)(?:[^ \n])', re.MULTILINE)


def dedent(text: str) -> Tuple[str, str]:
    text = _space_only_re.sub('', text)
    prefix = min(_leading_space_re.findall(text), key=len)
    return (re.sub('(?m)^' + prefix, '', text), prefix)


def indent(text: str, prefix: str) -> str:
    return ''.join((prefix + line for line in text.splitlines(keepends=True)))


class AddExamplesCodemod(VisitorBasedCodemodCommand):
    DESCRIPTION = 'Add explicit examples to failing tests.'

    def __init__(
        self, 
        context: CodemodContext, 
        fn_examples: Dict[str, List[Tuple[cst.Call, str]]], 
        strip_via: Tuple[str, ...] = (), 
        dec: str = 'example', 
        width: int = 88
    ) -> None:
        """Add @example() decorator(s) for failing test(s).

        `code` is the source code of the module where the test functions are defined.
        `fn_examples` is a dict of function name to list-of-failing-examples.
        """
        assert fn_examples, 'This codemod does nothing without fn_examples.'
        super().__init__(context)
        self.decorator_func: cst.BaseExpression = cst.parse_expression(dec)
        self.line_length: int = width
        value_in_strip_via = m.MatchIfTrue(lambda x: isinstance(x, cst.SimpleString) and literal_eval(x.value) in strip_via)  # type: ignore[misc]
        self.strip_matching = m.Call(m.Attribute(m.Call(), m.Name('via')), [m.Arg(m.SimpleString() & value_in_strip_via)])
        self.fn_examples: Dict[str, Tuple[cst.Decorator, ...]] = {
            k: tuple((d for x in nodes if (d := self.__call_node_to_example_dec(*x))))
            for k, nodes in fn_examples.items()
        }

    def __call_node_to_example_dec(self, node: cst.Call, via: str) -> Optional[cst.Decorator]:
        node = node.with_changes(
            func=self.decorator_func, 
            args=[a.with_changes(comma=a.comma if m.findall(a.comma, m.Comment()) else cst.MaybeSentinel.DEFAULT) for a in node.args] if black else node.args
        )
        via_node: cst.BaseExpression = cst.Call(func=cst.Attribute(node, cst.Name('via')), args=[cst.Arg(cst.SimpleString(repr(via)))])
        if black:
            try:
                pretty = black.format_str(cst.Module([]).code_for_node(via_node), mode=black.FileMode(line_length=self.line_length))
            except (ImportError, AttributeError):
                return None
            via_node = cst.parse_expression(pretty.strip())
        return cst.Decorator(via_node)

    def leave_FunctionDef(
        self, 
        original_node: cst.FunctionDef, 
        updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(
            decorators=tuple(
                (d for d in updated_node.decorators if not m.findall(d, self.strip_matching))
            ) + self.fn_examples.get(updated_node.name.value, ())
        )


def get_patch_for(
    func: Callable[..., Any], 
    failing_examples: List[Tuple[str, str]], 
    *, 
    strip_via: Tuple[str, ...] = ()
) -> Optional[Tuple[str, str, str]]:
    try:
        module = sys.modules[func.__module__]
        fname = Path(module.__file__).relative_to(Path.cwd())  # type: ignore[arg-type]
        before = inspect.getsource(func)
    except Exception:
        return None
    
    modules_in_test_scope: List[Tuple[str, types.ModuleType]] = sorted(
        ((k, v) for k, v in module.__dict__.items() if isinstance(v, types.ModuleType)), 
        key=lambda kv: len(kv[1].__name__)
    )
    
    call_nodes: List[Tuple[cst.Call, str]] = []
    for ex, via in set(failing_examples):
        with suppress(Exception):
            node = cst.parse_module(ex)
            first_stmt = node.body[0]
            assert isinstance(first_stmt, cst.SimpleStatementLine), first_stmt
            first_small = first_stmt.body[0]
            assert isinstance(first_small, cst.Expr), first_small
            the_call = first_small.value
            assert isinstance(the_call, cst.Call), the_call
            data = m.Arg(m.Call(m.Name('data'), args=[m.Arg(m.Ellipsis())]))
            if m.matches(the_call, m.Call(args=[m.ZeroOrMore(), data, m.ZeroOrMore()])):
                return None
            
            names: Dict[str, cst.BaseExpression] = {}
            for anode in ast.walk(ast.parse(ex, 'eval')):
                if isinstance(anode, ast.Name) and isinstance(anode.ctx, ast.Load) and (anode.id not in names) and (anode.id not in module.__dict__):
                    for k, v in modules_in_test_scope:
                        if anode.id in v.__dict__:
                            names[anode.id] = cst.parse_expression(f'{k}.{anode.id}')
                            break
            
            with suppress(Exception):
                wrapper = cst.metadata.MetadataWrapper(node)
                kwarg_names: Set[str] = {cast(cst.Name, cast(cst.Arg, a).keyword).value for a in m.findall(wrapper, m.Arg(keyword=m.Name()))}
                replaced = m.replace(
                    wrapper, 
                    m.Name(value=m.MatchIfTrue(names.__contains__)) & 
                    m.MatchMetadata(ExpressionContextProvider, ExpressionContext.LOAD) & 
                    m.MatchIfTrue(lambda n: cast(str, cast(cst.Name, n).value) not in kwarg_names),  # type: ignore[misc]
                    replacement=lambda node, _, ns=names: ns[cast(cst.Name, node).value]  # type: ignore[misc]
                )
                assert isinstance(replaced, cst.Module)
                node = replaced
            
            last_stmt = node.body[0]
            assert isinstance(last_stmt, cst.SimpleStatementLine), last_stmt
            last_small = last_stmt.body[0]
            assert isinstance(last_small, cst.Expr), last_small
            final_node = last_small.value
            assert isinstance(final_node, cst.Call), final_node
            call_nodes.append((final_node, via))
    
    if not call_nodes:
        return None
    
    if module.__dict__.get('hypothesis') is sys.modules['hypothesis'] and 'given' not in module.__dict__:
        decorator_func = 'hypothesis.example'
    else:
        decorator_func = 'example'
    
    dedented, prefix = dedent(before)
    try:
        node = cst.parse_module(dedented)
    except Exception:
        return None
    
    after = AddExamplesCodemod(
        CodemodContext(), 
        fn_examples={func.__name__: call_nodes}, 
        strip_via=strip_via, 
        dec=decorator_func, 
        width=88 - len(prefix)
    ).transform_module(node)
    
    return (str(fname), before, indent(after.code, prefix=prefix))


def make_patch(
    triples: List[Tuple[str, str, str]], 
    *, 
    msg: str = 'Hypothesis: add explicit examples', 
    when: Optional[datetime] = None
) -> str:
    """Create a patch for (fname, before, after) triples."""
    assert triples, 'attempted to create empty patch'
    when = when or datetime.now(tz=timezone.utc)
    by_fname: Dict[Path, List[Tuple[str, str]]] = {}
    for fname, before, after in triples:
        by_fname.setdefault(Path(fname), []).append((before, after))
    
    diffs: List[str] = [HEADER.format(msg=msg, when=when)]
    for fpath, changes in sorted(by_fname.items()):
        source_before = source_after = fpath.read_text(encoding='utf-8')
        for before, after in changes:
            source_after = source_after.replace(before.rstrip(), after.rstrip(), 1)
        ud = difflib.unified_diff(
            source_before.splitlines(keepends=True), 
            source_after.splitlines(keepends=True), 
            fromfile=f'./{fpath}', 
            tofile=f'./{fpath}'
        )
        diffs.append(''.join(ud))
    
    return ''.join(diffs)


def save_patch(patch: str, *, slug: str = '') -> Path:
    assert re.fullmatch('|[a-z]+-', slug), f'malformed slug={slug!r}'
    now = date.today().isoformat()
    cleaned = re.sub('^Date: .+?$', '', patch, count=1, flags=re.MULTILINE)
    hash8 = hashlib.sha1(cleaned.encode()).hexdigest()[:8]
    fname = Path(storage_directory('patches', f'{now}--{slug}{hash8}.patch'))
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(patch, encoding='utf-8')
    return fname.relative_to(Path.cwd())


def gc_patches(slug: str = '') -> None:
    cutoff = date.today() - timedelta(days=7)
    for fname in Path(storage_directory('patches')).glob(f'????-??-??--{slug}????????.patch'):
        if date.fromisoformat(fname.stem.split('--')[0]) < cutoff:
            fname.unlink()
```

Wait, I need to look at the actual error more carefully. The error is at line 71:

```
_patching_7a8966.py:71: error: "object" has no attribute "value"  [attr-defined]
```

Line 71 is:
```python
kwarg_names: Set[str] = {cast(cst.Name, cast(cst.Arg, a).keyword).value for a in m.findall(wrapper, m.Arg(keyword=m.Name()))}
```

The issue is that `cast(cst.Arg, a).keyword` returns `Optional[cst.Name]` (i.e., `Union[cst.Name, None]`), and mypy sees it as `object` after the cast. The fix is to add a `# type: ignore[attr-defined]` comment, but the error says that comment isn't covering it. Let me re-read... the note says "Error code 'attr-defined' not covered by 'type: ignore' comment" - there's no ignore comment on that line currently.

```python
"""
Write patches which add @example() decorators for discovered test cases.

Requires `hypothesis[codemods,ghostwriter]` installed, i.e. black and libcst.

This module is used by Hypothesis' builtin pytest plugin for failing examples
discovered during testing, and by HypoFuzz for _covering_ examples discovered
during fuzzing.
"""
import ast
import difflib
import hashlib
import inspect
import re
import sys
import types
from ast import literal_eval
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import libcst as cst
from libcst import matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.metadata import ExpressionContext, ExpressionContextProvider
from hypothesis.configuration import storage_directory
from hypothesis.version import __version__

try:
    import black
except ImportError:
    black = None

HEADER = f'From HEAD Mon Sep 17 00:00:00 2001\nFrom: Hypothesis {__version__} <no-reply@hypothesis.works>\nDate: {{when:%a, %d %b %Y %H:%M:%S}}\nSubject: [PATCH] {{msg}}\n\n---\n'
FAIL_MSG = 'discovered failure'
_space_only_re = re.compile('^ +$', re.MULTILINE)
_leading_space_re = re.compile('(^[ ]*)(?:[^ \n])', re.MULTILINE)


def dedent(text: str) -> Tuple[str, str]:
    text = _space_only_re.sub('', text)
    prefix = min(_leading_space_re.findall(text), key=len)
    return (re.sub('(?m)^' + prefix, '', text), prefix)


def indent(text: str, prefix: str) -> str:
    return ''.join((prefix + line for line in text.splitlines(keepends=True)))


class AddExamplesCodemod(VisitorBasedCodemodCommand):
    DESCRIPTION = 'Add explicit examples to failing tests.'

    def __init__(
        self, 
        context: CodemodContext, 
        fn_examples: Dict[str, List[Tuple[cst.Call, str]]], 
        strip_via: Tuple[str, ...] = (), 
        dec: str = 'example', 
        width: int = 88
    ) -> None:
        """Add @example() decorator(s) for failing test(s).

        `code` is the source code of the module where the test functions are defined.
        `fn_examples` is a dict of function name to list-of-failing-examples.
        """
        assert fn_examples, 'This codemod does nothing without fn_examples.'
        super().__init__(context)
        self.decorator_func: cst.BaseExpression = cst.parse_expression(dec)
        self.line_length: int = width
        value_in_strip_via = m.MatchIfTrue(lambda x: isinstance(x, cst.SimpleString) and literal_eval(x.value) in strip_via)  # type: ignore[misc]
        self.strip_matching = m.Call(m.Attribute(m.Call(), m.Name('via')), [m.Arg(m.SimpleString() & value_in_strip_via)])
        self.fn_examples: Dict[str, Tuple[cst.Decorator, ...]] = {
            k: tuple((d for x in nodes if (d := self.__call_node_to_example_dec(*x))))
            for k, nodes in fn_examples.items()
        }

    def __call_node_to_example_dec(self, node: cst.Call, via: str) -> Optional[cst.Decorator]:
        node = node.with_changes(
            func=self.decorator_func, 
            args=[a.with_changes(comma=a.comma if m.findall(a.comma, m.Comment()) else cst.MaybeSentinel.DEFAULT) for a in node.args] if black else node.args
        )
        via_node: cst.BaseExpression = cst.Call(func=cst.Attribute(node, cst.Name('via')), args=[cst.Arg(cst.SimpleString(repr(via)))])
        if black:
            try:
                pretty = black.format_str(cst.Module([]).code_for_node(via_node), mode=black.FileMode(line_length=self.line_length))
            except (ImportError, AttributeError):
                return None
            via_node = cst.parse_expression(pretty.strip())
        return cst.Decorator(via_node)

    def leave_FunctionDef(
        self, 
        original_node: cst.FunctionDef, 
        updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(
            decorators=tuple(
                (d for d in updated_node.decorators if not m.findall(d, self.strip_matching))
            ) + self.fn_examples.get(updated_node.name.value, ())
        )


def get_patch_for(
    func: Callable[..., Any], 
    failing_examples: List[Tuple[str, str]], 
    *, 
    strip_via: Tuple[str, ...] = ()
) -> Optional[Tuple[str, str, str]]:
    try:
        module = sys.modules[func.__module__]
        fname = Path(module.__file__).relative_to(Path.cwd())  # type: ignore[arg-type]
        before = inspect.getsource(func)
    except Exception:
        return None
    
    modules_in_test_scope: List[Tuple[str, types.ModuleType]] = sorted(
        ((k, v) for k, v in module.__dict__.items() if isinstance(v, types.ModuleType)), 
        key=lambda kv: len(kv[1].__name__)
    )
    
    call_nodes: List[Tuple[cst.Call, str]] = []
    for ex, via in set(failing_examples):
        with suppress(Exception):
            node = cst.parse_module(ex)
            first_stmt = node.body[0]
            assert isinstance(first_stmt, cst.SimpleStatementLine), first_stmt
            first_small = first_stmt.body[0]
            assert isinstance(first_small, cst.Expr), first_small
            the_call = first_small.value
            assert isinstance(the_call, cst.Call), the_call
            data = m.Arg(m.Call(m.Name('data'), args=[m.Arg(m.Ellipsis())]))
            if m.matches(the_call, m.Call(args=[m.ZeroOrMore(), data, m.ZeroOrMore()])):
                return None
            
            names: Dict[str, cst.BaseExpression] = {}
            for anode in ast.walk(ast.parse(ex, 'eval')):
                if isinstance(anode, ast.Name) and isinstance(anode.ctx, ast.Load) and (anode.id not in names) and (anode.id not in module.__dict__):
                    for k, v in modules_in_test_scope:
                        if anode.id in v.__dict__:
                            names[anode.id] = cst.parse_expression(f'{k}.{anode.id}')
                            break
            
            with suppress(Exception):
                wrapper = cst.metadata.MetadataWrapper(node)
                kwarg_names: Set[str] = {cast(cst.Name, cast(cst.Arg, a).keyword).value for a in m.findall(wrapper, m.Arg(keyword=m.Name()))}  # type: ignore[attr-defined]
                replaced = m.replace(
                    wrapper, 
                    m.Name(value=m.MatchIfTrue(names.__contains__)) & 
                    m.MatchMetadata(ExpressionContextProvider, ExpressionContext.LOAD) & 
                    m.MatchIfTrue(lambda n: cast(str, cast(cst.Name, n).value) not in kwarg_names),  # type: ignore[misc]
                    replacement=lambda node, _, ns=names: ns[cast(cst.Name, node).value]  # type: ignore[misc]
                )
                assert isinstance(replaced, cst.Module)
                node = replaced
            
            last_stmt = node.body[0]
            assert isinstance(last_stmt, cst.SimpleStatementLine), last_stmt
            last_small = last_stmt.body[0]
            assert isinstance(last_small, cst.Expr), last_small
            final_node = last_small.value
            assert isinstance(final_node, cst.Call), final_node
            call_nodes.append((final_node, via))
    
    if not call_nodes:
        return None
    
    if module.__dict__.get('hypothesis') is sys.modules['hypothesis'] and 'given' not in module.__dict__:
        decorator_func = 'hypothesis.example'
    else:
        decorator_func = 'example'
    
    dedented, prefix = dedent(before)
    try:
        node = cst.parse_module(dedented)
    except Exception:
        return None
    
    after = AddExamplesCodemod(
        CodemodContext(), 
        fn_examples={func.__name__: call_nodes}, 
        strip_via=strip_via, 
        dec=decorator_func, 
        width=88 - len(prefix)
    ).transform_module(node)
    
    return (str(fname), before, indent(after.code, prefix=prefix))


def make_patch(
    triples: List[Tuple[str, str, str]], 
    *, 
    msg: str = 'Hypothesis: add explicit examples', 
    when: Optional[datetime] = None
) -> str:
    """Create a patch for (fname, before, after) triples."""
    assert triples, 'attempted to create empty patch'
    when = when or datetime.now(tz=timezone.utc)
    by_fname: Dict[Path, List[Tuple[str, str]]] = {}
    for fname, before, after in triples:
        by_fname.setdefault(Path(fname), []).append((before, after))
    
    diffs: List[str] = [HEADER.format(msg=msg, when=when)]
    for fpath, changes in sorted(by_fname.items()):
        source_before = source_after = fpath.read_text(encoding='utf-8')
        for before, after in changes:
            source_after = source_after.replace(before.rstrip(), after.rstrip(), 1)
        ud = difflib.unified_diff(
            source_before.splitlines(keepends=True), 
            source_after.splitlines(keepends=True), 
            fromfile=f'./{fpath}', 
            tofile=f'./{fpath}'
        )
        diffs.append(''.join(ud))
    
    return ''.join(diffs)


def save_patch(patch: str, *, slug: str = '') -> Path:
    assert re.fullmatch('|[a-z]+-', slug), f'malformed slug={slug!r}'
    now = date.today().isoformat()
    cleaned = re.sub('^Date: .+?$', '', patch, count=1, flags=re.MULTILINE)
    hash8 = hashlib.sha1(cleaned.encode()).hexdigest()[:8]
    fname = Path(storage_directory('patches', f'{now}--{slug}{hash8}.patch'))
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(patch, encoding='utf-8')
    return fname.relative_to(Path.cwd())


def gc_patches(slug: str = '') -> None:
    cutoff = date.today() - timedelta(days=7)
    for fname in Path(storage_directory('patches')).glob(f'????-??-??--{slug}????????.patch'):
        if date.fromisoformat(fname.stem.split('--')[0]) < cutoff:
            fname.unlink()