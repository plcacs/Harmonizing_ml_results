import json
import os
import csv
import re
from typing import Dict, List, Tuple


def load_json_data(filepath: str) -> Dict:
    """Load JSON data from file"""
    with open(filepath, "r") as f:
        return json.load(f)


def has_optional_type(type_str: str) -> bool:
    """Check if a type string contains Optional or Union[..., None]"""
    if not type_str or type_str.strip() == "":
        return False
    
    type_str = type_str.strip("'\"")
    
    # Check for Optional
    if "Optional" in type_str:
        return True
    
    # Check for Union with None
    if "Union" in type_str and "None" in type_str:
        return True
    
    # Check for | None pattern
    if " | None" in type_str or "| None" in type_str:
        return True
    
    return False


def extract_optional_differences(data: Dict) -> List[Tuple]:
    """Extract parameters where either human or LLM has optional type"""
    differences = []
    
    for filename, funcs in data.items():
        for func_sig, params in funcs.items():
            for param in params:
                human_type = param.get("Human", "")
                llm_type = param.get("LLM", "")
                param_name = param.get("param_name", "")
                category = param.get("category", "")
                
                # Check if either human or LLM has optional type
                human_optional = has_optional_type(human_type)
                llm_optional = has_optional_type(llm_type)
                
                if human_optional or llm_optional:
                    # Extract function and class from function signature
                    func_class = func_sig
                    
                    # Create parameter name/return field with category information
                    param_return_info = f"{category}:{param_name}" if param_name else category
                    
                    differences.append((
                        filename,
                        func_class,
                        param_return_info,
                        human_type,
                        llm_type
                    ))
    
    return differences


def generate_csv(differences: List[Tuple], output_filename: str):
    """Generate CSV file with optional differences"""
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'File name',
            'Function@class',
            'Parameter name/return',
            'Human annotation',
            'LLM annotation'
        ])
        
        # Write data
        for row in differences:
            writer.writerow(row)


def main():
    """Main function to process all semantic comparison files"""
    # Get all semantic comparison files
    json_files = [
        f
        for f in os.listdir(".")
        if f.startswith("type_comparison_semantic_") and f.endswith(".json")
    ]

    if not json_files:
        print("No semantic comparison JSON files found!")
        return

    print(f"Found {len(json_files)} semantic comparison files")

    for json_file in json_files:
        llm_name = json_file.replace("type_comparison_semantic_", "").replace(
            ".json", ""
        )
        output_filename = f"optional_differences_{llm_name}.csv"

        print(f"\n{'='*80}")
        print(f"PROCESSING: {json_file}")
        print(f"OUTPUT: {output_filename}")
        print(f"{'='*80}")

        try:
            data = load_json_data(json_file)
            differences = extract_optional_differences(data)
            
            print(f"Found {len(differences)} parameters with optional differences")
            
            generate_csv(differences, output_filename)
            print(f"CSV file saved to: {output_filename}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    main() 