#!/usr/bin/env python3
"""
Map basenames in selected_500_files.json (untyped_benchmarks-style names) to absolute
paths listed in model_results/cloned_repos/filtered_python_files.json.

Matching rule matches Generate_no_type_version.py: for each repo file, the untyped
basename would be "{stem}_{md5(untyped_ast_unparse)[:6]}.py" where stem is the
source file basename without .py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)


class RemoveTypeHints(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args + getattr(node.args, "kwonlyargs", []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node):
        if node.value:
            new_node = ast.Assign(
                targets=[node.target],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            return ast.copy_location(new_node, node)
        return None

    def visit_arg(self, node):
        node.annotation = None
        return node


def untyped_hash6(code: str, filename_hint: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code, filename=filename_hint)
    transformer = RemoveTypeHints()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    try:
        untyped_code = ast.unparse(tree)
    except AttributeError:
        import astunparse

        untyped_code = astunparse.unparse(tree)
    return hashlib.md5(untyped_code.encode("utf-8")).hexdigest()[:6]


_BENCHMARK_NAME = re.compile(r"^(.+)_([0-9a-f]{6})\.py$", re.IGNORECASE)


def parse_selected_basename(name: str) -> tuple[str, str] | None:
    m = _BENCHMARK_NAME.match(name)
    if not m:
        return None
    return m.group(1), m.group(2).lower()


def flatten_filtered(data: dict) -> list[str]:
    out: list[str] = []
    for paths in data.values():
        if isinstance(paths, list):
            out.extend(paths)
    return out


def strip_common_prefix(abs_path: str, prefix: str) -> str:
    """Drop a shared root from absolute paths; return posix-style relative remainder."""
    if not prefix or not abs_path:
        return abs_path
    try:
        rel = Path(abs_path).relative_to(Path(prefix))
        return rel.as_posix()
    except ValueError:
        return abs_path


def expected_benchmark_basename(repo_path: str, code: str) -> str | None:
    """Basename as written by Generate_no_type_version.process_py_file."""
    try:
        h = untyped_hash6(code, repo_path)
    except (SyntaxError, UnicodeDecodeError):
        return None
    stem = Path(repo_path).stem
    return f"{stem}_{h}.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path(__file__).resolve().parent / "selected_500_files.json",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "model_results"
        / "cloned_repos"
        / "filtered_python_files.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "selected_500_to_repo_paths.json",
    )
    parser.add_argument(
        "--strip-prefix",
        default=r"D:\Projects\Datasets\many-types-4-py-dataset\data",
        help="Remove this prefix from paths in output (use empty string to keep absolute).",
    )
    parser.add_argument(
        "--keep-absolute",
        action="store_true",
        help="Write full absolute paths (ignore --strip-prefix).",
    )
    args = parser.parse_args()

    strip_prefix = "" if args.keep_absolute else (args.strip_prefix or "")

    def out_path(p: str) -> str:
        return p if not strip_prefix else strip_common_prefix(p, strip_prefix)

    with open(args.selected, encoding="utf-8") as f:
        selected_doc = json.load(f)
    selected_names = selected_doc.get("files", [])
    if not selected_names:
        raise SystemExit("selected JSON has no 'files' list")

    with open(args.filtered, encoding="utf-8") as f:
        filtered_doc = json.load(f)
    all_paths = flatten_filtered(filtered_doc)

    stems_needed: set[str] = set()
    bad_name: list[str] = []
    for name in selected_names:
        parsed = parse_selected_basename(name)
        if parsed:
            stems_needed.add(parsed[0])
        else:
            bad_name.append(name)

    # basename -> repo paths (only for stems that appear in the selection)
    basename_hits: dict[str, list[str]] = defaultdict(list)
    missing_file: list[str] = []
    parse_skipped: list[str] = []

    for repo_path in all_paths:
        stem = Path(repo_path).stem
        if stem not in stems_needed:
            continue
        if not os.path.isfile(repo_path):
            missing_file.append(repo_path)
            continue
        try:
            with open(repo_path, encoding="utf-8") as rf:
                code = rf.read()
        except OSError:
            parse_skipped.append(repo_path)
            continue
        bench = expected_benchmark_basename(repo_path, code)
        if bench is None:
            parse_skipped.append(repo_path)
            continue
        if repo_path not in basename_hits[bench]:
            basename_hits[bench].append(repo_path)

    mapping: dict[str, str | None] = {}
    ambiguous: list[dict] = []
    no_match: list[str] = []

    for name in selected_names:
        if name in bad_name:
            mapping[name] = None
            continue
        hits = basename_hits.get(name, [])
        if len(hits) == 1:
            mapping[name] = out_path(hits[0])
        elif len(hits) == 0:
            no_match.append(name)
            mapping[name] = None
        else:
            ambiguous.append(
                {"selected": name, "repo_paths": [out_path(x) for x in hits]}
            )
            mapping[name] = None

    out_doc = {
        "selected_json": str(args.selected),
        "filtered_json": str(args.filtered),
        "path_prefix_removed": strip_prefix or None,
        "resolved": sum(1 for v in mapping.values() if v),
        "total": len(selected_names),
        "mapping": mapping,
        "issues": {
            "bad_basename_pattern": bad_name,
            "no_stem_or_hash_match": no_match,
            "ambiguous_multiple_repo_hits": ambiguous,
            "repo_path_missing_on_disk_sample": missing_file[:20],
            "parse_skipped_sample": parse_skipped[:30],
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    print(f"Wrote {args.output}")
    print(f"Resolved {out_doc['resolved']} / {out_doc['total']}")
    if bad_name:
        print(f"Bad name pattern: {len(bad_name)}")
    if no_match:
        print(f"No match: {len(no_match)}")
    if ambiguous:
        print(f"Ambiguous: {len(ambiguous)}")
    if missing_file:
        print(f"Missing on disk (first 20 logged): {len(missing_file)} paths")


if __name__ == "__main__":
    main()
