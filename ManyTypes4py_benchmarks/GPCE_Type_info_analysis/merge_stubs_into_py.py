"""
Merge type annotations from .pyi stub files into .py source files.

For each matched pair (foo.py + foo.pyi):
  1. Parse both files into ASTs.
  2. Copy parameter annotations and return annotations from stub functions
     into the corresponding .py functions (matched by qualified name).
  3. Collect all imports from the stub and merge any missing ones into the .py.
  4. Write the merged result to the output directory.
"""

import ast
import os
import sys
import glob
import shutil
import textwrap
from collections import defaultdict


# ---------------------------------------------------------------------------
# Import extraction helpers
# ---------------------------------------------------------------------------

def _extract_imports(tree):
    """Return a list of import AST nodes from the top level of a module."""
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    return imports


def _import_key(node):
    """Unique key for deduplication of import statements."""
    if isinstance(node, ast.Import):
        return ("import", tuple(sorted(a.name for a in node.names)))
    else:
        module = node.module or ""
        return ("from", module, node.level)


def _names_from_import(node):
    """Return set of imported names from an import node."""
    return {(a.asname or a.name) for a in node.names}


def _merge_imports(py_lines, py_tree, stub_tree):
    """
    Merge imports from stub into py source lines.
    Returns the new source lines with missing imports added.
    """
    py_imports = _extract_imports(py_tree)
    stub_imports = _extract_imports(stub_tree)

    # Build map: (kind, module, level) -> set of names already in .py
    py_from_map = {}
    for node in py_imports:
        if isinstance(node, ast.ImportFrom):
            key = (node.module or "", node.level)
            if key not in py_from_map:
                py_from_map[key] = {"node": node, "names": set()}
            py_from_map[key]["names"].update(a.name for a in node.names)

    py_plain_imports = set()
    for node in py_imports:
        if isinstance(node, ast.Import):
            for alias in node.names:
                py_plain_imports.add(alias.name)

    new_import_lines = []

    for snode in stub_imports:
        if isinstance(snode, ast.Import):
            for alias in snode.names:
                if alias.name not in py_plain_imports:
                    if alias.asname:
                        new_import_lines.append(f"import {alias.name} as {alias.asname}")
                    else:
                        new_import_lines.append(f"import {alias.name}")
        elif isinstance(snode, ast.ImportFrom):
            key = (snode.module or "", snode.level)
            existing_names = py_from_map.get(key, {}).get("names", set())
            missing = []
            for alias in snode.names:
                if alias.name not in existing_names:
                    missing.append(alias)
            if missing:
                if key in py_from_map:
                    # Extend the existing from-import line in-place
                    orig_node = py_from_map[key]["node"]
                    line_idx = orig_node.end_lineno - 1
                    orig_line = py_lines[line_idx]

                    new_names_str = ", ".join(
                        f"{a.name} as {a.asname}" if a.asname else a.name
                        for a in missing
                    )

                    # Handle multi-line imports ending with ')'
                    stripped = orig_line.rstrip()
                    if stripped.endswith(")"):
                        py_lines[line_idx] = stripped[:-1].rstrip() + ", " + new_names_str + ")\n"
                    else:
                        py_lines[line_idx] = stripped + ", " + new_names_str + "\n"

                    py_from_map[key]["names"].update(a.name for a in missing)
                else:
                    prefix = "." * snode.level + (snode.module or "")
                    names_str = ", ".join(
                        f"{a.name} as {a.asname}" if a.asname else a.name
                        for a in missing
                    )
                    new_import_lines.append(f"from {prefix} import {names_str}")
                    py_from_map[key] = {"node": snode, "names": {a.name for a in snode.names}}

    if new_import_lines:
        # Find the last import line in the .py to insert after
        last_import_line = 0
        for node in py_imports:
            if node.end_lineno and node.end_lineno > last_import_line:
                last_import_line = node.end_lineno

        insert_at = last_import_line
        for line in reversed(new_import_lines):
            py_lines.insert(insert_at, line + "\n")

    return py_lines


# ---------------------------------------------------------------------------
# Function matching & annotation transfer
# ---------------------------------------------------------------------------

def _build_func_map(tree):
    """
    Build a dict mapping qualified function names to their AST nodes.
    Qualified = "ClassName.method" or just "function" for top-level.
    """
    func_map = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_map[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_map[f"{node.name}.{child.name}"] = child

    return func_map


def _annotation_to_source(node):
    """Convert an annotation AST node back to source code string."""
    return ast.unparse(node)


def _apply_annotations(py_lines, py_tree, stub_tree):
    """
    Transfer annotations from stub functions into py source lines.
    Works by modifying the `def` signature lines in py_lines.
    """
    py_funcs = _build_func_map(py_tree)
    stub_funcs = _build_func_map(stub_tree)

    # Process in reverse line order so edits don't shift later line numbers
    edits = []

    for qname, stub_fn in stub_funcs.items():
        py_fn = py_funcs.get(qname)
        if py_fn is None:
            continue

        # Build the new signature from the py function body but with stub annotations
        new_sig = _build_annotated_signature(py_fn, stub_fn, py_lines)
        if new_sig is not None:
            edits.append((py_fn.lineno - 1, py_fn.end_lineno, new_sig))

    # Sort edits by start line descending so we can apply without offset issues
    edits.sort(key=lambda e: e[0], reverse=True)

    for start, _end, new_sig in edits:
        # Find the end of the def line (could be multi-line, ends with ':')
        def_end = start
        for idx in range(start, len(py_lines)):
            if py_lines[idx].rstrip().endswith(":"):
                def_end = idx
                break

        # Replace just the def line(s), keep the body
        py_lines[start:def_end + 1] = [new_sig + "\n"]

    return py_lines


def _build_annotated_signature(py_fn, stub_fn, py_lines):
    """
    Build a new def line with annotations from stub_fn applied to py_fn's
    structure, preserving decorators and the original indentation.
    """
    # Get indentation from the original def line
    orig_line = py_lines[py_fn.lineno - 1]
    indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]

    async_prefix = "async " if isinstance(py_fn, ast.AsyncFunctionDef) else ""

    # Build parameter list with annotations from stub
    py_args = py_fn.args
    stub_args = stub_fn.args

    # Map stub param names to their annotations
    stub_ann_map = {}
    all_stub_params = (
        list(getattr(stub_args, "posonlyargs", []))
        + list(stub_args.args)
        + list(stub_args.kwonlyargs)
    )
    if stub_args.vararg:
        all_stub_params.append(stub_args.vararg)
    if stub_args.kwarg:
        all_stub_params.append(stub_args.kwarg)

    for arg in all_stub_params:
        if arg.annotation:
            stub_ann_map[arg.arg] = _annotation_to_source(arg.annotation)

    # Build param strings from py_fn structure
    params = []

    # Positional-only args
    posonlyargs = getattr(py_args, "posonlyargs", [])
    num_posonly = len(posonlyargs)
    # Defaults for positional args: the last N positional args have defaults
    # defaults covers args (not posonlyargs separately in older Python)
    num_args = len(py_args.args)
    num_defaults = len(py_args.defaults)

    all_positional = list(posonlyargs) + list(py_args.args)
    total_positional = len(all_positional)
    # defaults align to the end of all_positional
    default_offset = total_positional - num_defaults

    for i, arg in enumerate(all_positional):
        s = arg.arg
        if s in stub_ann_map:
            s += f": {stub_ann_map[s]}"
        if i >= default_offset:
            default_node = py_args.defaults[i - default_offset]
            default_str = _annotation_to_source(default_node)
            s += f" = {default_str}"
        params.append(s)
        if num_posonly > 0 and i == num_posonly - 1:
            params.append("/")

    # *args
    if py_args.vararg:
        s = f"*{py_args.vararg.arg}"
        if py_args.vararg.arg in stub_ann_map:
            s += f": {stub_ann_map[py_args.vararg.arg]}"
        params.append(s)
    elif py_args.kwonlyargs:
        params.append("*")

    # keyword-only args
    kw_defaults = py_args.kw_defaults
    for i, arg in enumerate(py_args.kwonlyargs):
        s = arg.arg
        if s in stub_ann_map:
            s += f": {stub_ann_map[s]}"
        if kw_defaults[i] is not None:
            default_str = _annotation_to_source(kw_defaults[i])
            s += f" = {default_str}"
        params.append(s)

    # **kwargs
    if py_args.kwarg:
        s = f"**{py_args.kwarg.arg}"
        if py_args.kwarg.arg in stub_ann_map:
            s += f": {stub_ann_map[py_args.kwarg.arg]}"
        params.append(s)

    params_str = ", ".join(params)

    # Return annotation
    ret = ""
    if stub_fn.returns:
        ret = f" -> {_annotation_to_source(stub_fn.returns)}"

    return f"{indent}{async_prefix}def {py_fn.name}({params_str}){ret}:"


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def merge_stub_into_py(py_path, stub_path):
    """Merge annotations from stub_path into py_path, return merged source."""
    with open(py_path, "r", encoding="utf-8") as f:
        py_source = f.read()
    with open(stub_path, "r", encoding="utf-8") as f:
        stub_source = f.read()

    # Strip markdown fences if present (LLM artifacts)
    stub_source = _strip_markdown_fences(stub_source)

    try:
        py_tree = ast.parse(py_source)
    except SyntaxError:
        print(f"  SKIP (py SyntaxError): {py_path}")
        return None
    try:
        stub_tree = ast.parse(stub_source)
    except SyntaxError:
        print(f"  SKIP (stub SyntaxError): {stub_path}")
        return None

    py_lines = py_source.splitlines(keepends=True)

    # Step 1: apply annotations (do this first as it changes line count)
    py_lines = _apply_annotations(py_lines, py_tree, stub_tree)

    # Re-parse after annotation edits to get correct line numbers for import merging
    try:
        new_tree = ast.parse("".join(py_lines))
    except SyntaxError:
        print(f"  WARNING: SyntaxError after annotation merge for {py_path}, skipping import merge")
        return "".join(py_lines)

    # Step 2: merge imports
    py_lines = _merge_imports(py_lines, new_tree, stub_tree)

    return "".join(py_lines)


def _strip_markdown_fences(text):
    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "".join(lines)


def main():
    # `py_directory` is the stub run root that contains .pyi files.
    py_directory = os.path.join("..", "gpt5_1_infer_stub_run")
    untyped_benchmarks_directory = os.path.join("..", "untyped_benchmarks")
    output_directory = os.path.join("..", "gpt5_1_infer_stub_run", "merged")

    if len(sys.argv) >= 3:
        py_directory = sys.argv[1]
        output_directory = sys.argv[2]
    if len(sys.argv) >= 4:
        untyped_benchmarks_directory = sys.argv[3]

    os.makedirs(output_directory, exist_ok=True)

    # Recursively find all stub files under the provided stub-run directory.
    stub_files = glob.glob(os.path.join(py_directory, "**", "*.pyi"), recursive=True)
    total = 0
    merged = 0
    skipped = 0

    for stub_path in sorted(stub_files):
        rel_stub_path = os.path.relpath(stub_path, py_directory)
        rel_py_path = os.path.splitext(rel_stub_path)[0] + ".py"
        py_path = os.path.join(untyped_benchmarks_directory, rel_py_path)

        if not os.path.exists(py_path):
            continue

        total += 1
        print(f"[{total}] Merging: {rel_py_path}")

        result = merge_stub_into_py(py_path, stub_path)
        if result is None:
            skipped += 1
            continue

        out_path = os.path.join(output_directory, rel_py_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        merged += 1

    print(f"\nDone. {merged} files merged, {skipped} skipped, out of {total} matched stub/source pairs.")
    print(f"Output directory: {os.path.abspath(output_directory)}")


if __name__ == "__main__":
    main()
