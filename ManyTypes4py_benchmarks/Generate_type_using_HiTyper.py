import os
import subprocess

# Path to the project root
project_root = "untyped_benchmarks"
# Output directory where HiTyper saves results
output_dir = "hityper_outputs"
# Depth of prediction (e.g., top-1)
top_k = "3"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Traverse and collect all .py files
py_files = []
for root, _, files in os.walk(project_root):
    for file in files:
        if file.endswith(".py"):
            py_files.append(os.path.join(root, file))

# Run HiTyper infer for each file
for py_file in py_files:
    print(f"Running hityper infer on: {py_file}")
    subprocess.run([
        "hityper", "infer",
        "-s", py_file,
        "-p", project_root,
        "-d", output_dir,
        "-n", top_k,
        "-t"
    ])
