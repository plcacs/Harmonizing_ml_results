import ast
import os
import random

BASE_DIR = os.path.dirname(__file__)
SELECTED_DIR = os.path.join(BASE_DIR, "selected_files")
SEED = 42

PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
FOLDER_NAMES = {
    10: "ten_percent_typed",
    20: "twenty_percent_typed",
    30: "thirty_percent_typed",
    40: "forty_percent_typed",
    50: "fifty_percent_typed",
    60: "sixty_percent_typed",
    70: "seventy_percent_typed",
    80: "eighty_percent_typed",
    90: "ninety_percent_typed",
}

for pct in PERCENTAGES:
    os.makedirs(os.path.join(BASE_DIR, FOLDER_NAMES[pct]), exist_ok=True)


def collect_annotated_params(tree):
    """Return list of ast.arg nodes that have annotations (excluding self)."""
    params = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)
        for arg in all_args:
            if arg.arg == "self":
                continue
            if arg.annotation is not None:
                params.append(arg)
    return params


def remove_annotations(source, params_to_remove):
    """Remove type annotations from the given params in source code.
    Operates on raw source to preserve comments and formatting."""
    sorted_params = sorted(
        params_to_remove,
        key=lambda a: (a.lineno, a.col_offset),
        reverse=True,
    )
    lines = source.splitlines(keepends=True)

    for arg in sorted_params:
        start_line = arg.lineno - 1
        start_col = arg.col_offset + len(arg.arg)
        end_line = arg.end_lineno - 1
        end_col = arg.end_col_offset

        if start_line == end_line:
            line = lines[start_line]
            lines[start_line] = line[:start_col] + line[end_col:]
        else:
            first = lines[start_line]
            last = lines[end_line]
            lines[start_line] = first[:start_col] + last[end_col:]
            for i in range(start_line + 1, end_line + 1):
                lines[i] = ""

    return "".join(lines)


files = sorted(f for f in os.listdir(SELECTED_DIR) if f.endswith(".py"))
print(f"Processing {len(files)} files across {len(PERCENTAGES)} percentages...\n")

skipped = 0
for idx, filename in enumerate(files):
    filepath = os.path.join(SELECTED_DIR, filename)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"  SKIP (SyntaxError): {filename}")
        skipped += 1
        continue

    annotated = collect_annotated_params(tree)
    if not annotated:
        print(f"  SKIP (no annotations): {filename}")
        skipped += 1
        continue

    total = len(annotated)

    for pct in PERCENTAGES:
        n_keep = round(total * pct / 100)

        rng = random.Random(f"{SEED}_{filename}_{pct}")
        indices = list(range(total))
        rng.shuffle(indices)
        to_remove = [annotated[i] for i in indices[n_keep:]]

        modified = remove_annotations(source, to_remove)

        out_path = os.path.join(BASE_DIR, FOLDER_NAMES[pct], filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(modified)

    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(files)}")

print(f"\nDone! Processed {len(files) - skipped} files, skipped {skipped}")
