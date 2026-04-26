import pandas as pd
from pathlib import Path

csv_dir = Path("./")

with pd.ExcelWriter("comparison_all_folders.xlsx", engine="xlsxwriter") as writer:
    for i in range(1, 18):
        csv_path = csv_dir / f"comparison_results_{i}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=f"Folder_{i}", index=False)
            print(f"Added Folder_{i}")

print("Done!")