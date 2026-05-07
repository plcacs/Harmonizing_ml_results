"""
Union imports from a 'run' directory into a 'merged' directory's files.
Usage: python union_imports.py <merged_dir> <run_dir>
Defaults to deepseek_3_stub_run/merged + deepseek_4_run if no args given.
"""

import ast
import os
import shutil
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")

if len(sys.argv) >= 3:
    MERGED_DIR = os.path.join(BASE, sys.argv[1])
    RUN4_DIR = os.path.join(BASE, sys.argv[2])
else:
    MERGED_DIR = os.path.join(BASE, "deepseek_3_stub_run", "merged")
    RUN4_DIR = os.path.join(BASE, "deepseek_4_run")

merged_parent = os.path.dirname(MERGED_DIR)
OUT_DIR = os.path.join(merged_parent, "merged_union_import")
os.makedirs(OUT_DIR, exist_ok=True)


def find_run4_file(filename):
    """Find a file in deepseek_4_run subdirectories."""
    for subdir in os.listdir(RUN4_DIR):
        path = os.path.join(RUN4_DIR, subdir, filename)
        if os.path.isfile(path):
            return path
    return None


def extract_import_lines(source):
    """Extract unique import statement strings from source code."""
    imports = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.add(ast.get_source_segment(source, node))
    imports.discard(None)
    return imports


def get_import_end_line(source):
    """Find the line number after the last top-level import."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    last_import_end = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = node.end_lineno
    return last_import_end


def process_file(filename):
    merged_path = os.path.join(MERGED_DIR, filename)
    run4_path = find_run4_file(filename)
    out_path = os.path.join(OUT_DIR, filename)

    if not run4_path:
        shutil.copy2(merged_path, out_path)
        return 0

    with open(merged_path, "r", encoding="utf-8", errors="replace") as f:
        merged_src = f.read()
    with open(run4_path, "r", encoding="utf-8", errors="replace") as f:
        run4_src = f.read()

    merged_imports = extract_import_lines(merged_src)
    run4_imports = extract_import_lines(run4_src)
    new_imports = run4_imports - merged_imports

    if not new_imports:
        shutil.copy2(merged_path, out_path)
        return 0

    insert_line = get_import_end_line(merged_src)
    lines = merged_src.splitlines(keepends=True)
    new_block = "\n".join(sorted(new_imports)) + "\n"
    lines.insert(insert_line, new_block)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return len(new_imports)


if __name__ == "__main__":
    files = sorted(os.listdir(MERGED_DIR))[:500]
    total_added = 0
    for i, fname in enumerate(files, 1):
        if not fname.endswith(".py"):
            continue
        added = process_file(fname)
        if added:
            print(f"[{i}/{len(files)}] {fname}: +{added} imports")
        total_added += added
    print(f"\nDone. Processed {len(files)} files. Total new imports added: {total_added}")
    print(f"Output: {OUT_DIR}")
