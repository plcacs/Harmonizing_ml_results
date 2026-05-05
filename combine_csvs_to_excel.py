"""
Combine all comparison_results_*.csv files into one Excel workbook with multiple sheets
"""

import pandas as pd
from pathlib import Path

def combine_csvs_to_excel(csv_dir: str, output_file: str = "comparison_results_all_folders.xlsx"):
    """
    Combine all comparison_results_*.csv files into a single Excel workbook.
    Each CSV becomes a separate sheet named by folder number.
    """
    
    csv_path = Path(csv_dir)
    
    # Get all comparison_results_*.csv files
    csv_files = sorted(csv_path.glob("comparison_results_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Create Excel writer
    output_path = csv_path / output_file
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for csv_file in csv_files:
            # Extract folder number from filename
            # e.g., "comparison_results_1.csv" -> "1"
            folder_num = csv_file.stem.split("_")[-1]
            sheet_name = f"Folder_{folder_num}"
            
            # Read CSV and write to Excel sheet
            df = pd.read_csv(csv_file)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ Sheet '{sheet_name}' - {len(df)} rows")
    
    print(f"\n✅ Excel file created: {output_path}")
    print(f"   Sheets: Folder_1 through Folder_18")


if __name__ == "__main__":
    # Use the comparison directory
    csv_directory = "./comparison_500_untyped_vs_deepseek_4_run_fresh"
    combine_csvs_to_excel(csv_directory)
