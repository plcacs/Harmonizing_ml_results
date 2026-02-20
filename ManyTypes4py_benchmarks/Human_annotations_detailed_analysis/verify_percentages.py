import ast
import os

BASE_DIR = os.path.dirname(__file__)
SELECTED_DIR = os.path.join(BASE_DIR, "selected_files")

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


def count_annotated_params(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    count = 0
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
                count += 1
    return count


files = sorted(f for f in os.listdir(SELECTED_DIR) if f.endswith(".py"))

for pct in PERCENTAGES:
    folder = os.path.join(BASE_DIR, FOLDER_NAMES[pct])
    mismatches = []
    total_files = 0

    for filename in files:
        orig_path = os.path.join(SELECTED_DIR, filename)
        var_path = os.path.join(folder, filename)
        if not os.path.exists(var_path):
            continue

        orig_count = count_annotated_params(orig_path)
        var_count = count_annotated_params(var_path)
        if orig_count is None or orig_count == 0:
            continue

        total_files += 1
        expected = round(orig_count * pct / 100)
        if var_count != expected:
            mismatches.append((filename, orig_count, expected, var_count))

    status = "OK" if not mismatches else f"{len(mismatches)} MISMATCHES"
    print(f"{pct:3d}% — {total_files} files checked — {status}")
    for name, orig, exp, got in mismatches[:3]:
        print(f"      {name}: orig={orig}, expected={exp}, got={got}")
